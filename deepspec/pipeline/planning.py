"""Persistent planning values. No Ray handles or GPU initialization belong here."""

import json
import math
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .runtime import PipelineError
from .schema import canonical_json, content_hash


@dataclass(frozen=True)
class Run:
    run_id: str
    namespace: str
    output_dir: str
    created_at: str

    @classmethod
    def create(cls, output_dir):
        path = Path(output_dir).resolve()
        # mkdir is the exclusive ownership operation, not an exists-then-write.
        try:
            path.mkdir(parents=True, exist_ok=False)
        except FileExistsError as error:
            raise PipelineError(
                "OUTPUT_EXISTS", "Use a new output directory", field_path="output_dir"
            ) from error
        run_id = uuid.uuid4().hex
        return cls(
            run_id,
            f"deepspec-{run_id}",
            str(path),
            datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class NodeFacts:
    _json: str

    @classmethod
    def from_dict(cls, value):
        for field in ("node_id", "gpus", "memory", "identities"):
            if field not in value:
                raise PipelineError(
                    "NODE_FACTS_MISSING",
                    f"Missing {field}",
                    field_path=f"nodes.{field}",
                )
        return cls(canonical_json(value))

    def to_dict(self):
        return json.loads(self._json)


@dataclass(frozen=True)
class InferenceReplica:
    replica_id: int
    tp: int
    worker_nodes: tuple[str, ...]
    core_node: str
    core_cpu_bundle: int
    writer_slot: int = 0

    def to_dict(self):
        return json.loads(canonical_json(asdict(self)))


@dataclass(frozen=True)
class TrainingParticipant:
    global_rank: int
    node_rank: int
    local_rank: int
    local_world_size: int
    node_id: str
    tp_rank: int
    dp_rank: int
    device_slot: int
    gpu_uuid: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TopologyPlan:
    _json: str

    @classmethod
    def freeze(cls, value):
        body = json.loads(canonical_json(value))
        body.pop("plan_hash", None)
        body["schema_version"] = 3
        body["plan_hash"] = content_hash(body)
        return cls(canonical_json(body))

    @classmethod
    def from_dict(cls, value):
        plan = cls.freeze(value)
        if plan.plan_hash != value.get("plan_hash"):
            raise PipelineError(
                "PLAN_HASH_MISMATCH",
                "Frozen plan hash does not match its contents",
                field_path="plan_hash",
            )
        return plan

    @property
    def plan_hash(self):
        return self.to_dict()["plan_hash"]

    def to_dict(self):
        return json.loads(self._json)


def resolve_node_facts(config, facts, *, now):
    """Resolve selectors uniquely and check capabilities without side effects."""
    selected = {}
    for spec in config["nodes"]:
        key, value = next(iter(spec["selector"].items()))
        matches = [n for n in facts if n.get("alive") and n.get(key) == value]
        if len(matches) != 1:
            raise PipelineError(
                "NODE_SELECTION",
                f"Expected one live node for {value}; found {len(matches)}",
                field_path=f"nodes.{spec['alias']}.selector",
                exit_code=4,
            )
        node = NodeFacts.from_dict(matches[0]).to_dict()
        if node["node_id"] in {n["node_id"] for n in selected.values()}:
            raise PipelineError(
                "DUPLICATE_NODE",
                "Aliases resolve to the same physical node",
                field_path="nodes",
            )
        age = now - node.get("request_sent_at", -math.inf)
        if not 0 <= age < config["timeouts_seconds"]["budget_snapshot"]:
            raise PipelineError(
                "NODE_FACTS_STALE",
                "Node facts must be freshly sampled",
                node_id=node["node_id"],
                field_path="nodes.memory",
                retryable=True,
                exit_code=4,
            )
        if not all(
            node.get("shared_paths", {}).get(k) is True
            for k in ("readable", "writable")
        ):
            raise PipelineError(
                "SHARED_PATH_UNAVAILABLE",
                "Shared inputs/output are not accessible",
                field_path="output_dir",
                exit_code=4,
            )
        if not all(
            node.get("capabilities", {}).get(k) is True
            for k in ("borrowed_pg", "cpu_core", "allocation_gate")
        ):
            raise PipelineError(
                "BACKEND_CAPABILITY_MISSING",
                "Native vLLM borrowed-PG, CPU-core and allocation-gate integration is required",
                node_id=node["node_id"],
                field_path="inference.backend",
            )
        identities = node["identities"]
        required = ("source", "dependencies", "model", "input")
        if any(not identities.get(k) for k in required):
            raise PipelineError(
                "ENVIRONMENT_IDENTITY_MISSING",
                "Required identity not measured",
                field_path="nodes.identities",
            )
        if selected and any(
            identities[k] != next(iter(selected.values()))["identities"][k]
            for k in required
        ):
            raise PipelineError(
                "ENVIRONMENT_IDENTITY_MISMATCH",
                "Nodes disagree on source, dependencies, model or input",
                field_path="nodes.identities",
            )
        uuids = [g["uuid"] for g in node["gpus"]]
        if len(set(uuids)) != len(uuids):
            raise PipelineError(
                "GPU_IDENTITY_CONFLICT",
                "Duplicate GPU inventory UUID",
                field_path="nodes.gpus",
            )
        selected[spec["alias"]] = node
    return selected


def build_plan(config, facts, inputs, *, run_id, now):
    from .memory import node_feature_budget
    from .schema import normalize_task_config
    from .topology import (
        expected_counts,
        planned_producer,
        planned_readers,
        training_rank_table,
    )

    config = normalize_task_config(config).to_dict()
    selected = resolve_node_facts(config, facts, now=now)
    samples = json.loads(canonical_json(inputs["batches"]))
    try:
        counts = expected_counts(config, sample_count=len(samples))
    except ValueError as error:
        raise PipelineError(
            "INPUT_PLAN_INVALID", str(error), field_path="data.samples"
        ) from error
    if (
        inputs["data_parallel_size"],
        inputs["global_batch_size"],
        inputs["gradient_accumulation_steps"],
    ) != (config["training"]["dp"], 4, counts["gas"]):
        raise PipelineError(
            "INPUT_PLAN_INVALID",
            "Native preparation changed training DP/batch/GAS",
            field_path="data.input_plan",
        )
    seen_ids = set()
    for position, sample in enumerate(samples):
        if sample["position"] != position or sample["id"] in seen_ids:
            raise PipelineError(
                "INPUT_PLAN_INVALID",
                "Samples must be unique and in frozen position order",
                field_path="data.samples",
            )
        seen_ids.add(sample["id"])
        if type(sample["nbytes"]) is not int or sample["nbytes"] <= 0:
            raise PipelineError(
                "INPUT_PLAN_INVALID",
                "Feature bytes must be a positive integer",
                field_path=f"data.samples.{position}.nbytes",
            )
        if "fields" in sample:
            element_sizes = {"int64": 8, "bool": 1, "float32": 4, "bfloat16": 2}
            try:
                actual_bytes = sum(
                    math.prod(f["shape"]) * element_sizes[f["dtype"]]
                    for f in sample["fields"].values()
                )
            except (KeyError, TypeError) as error:
                raise PipelineError(
                    "INPUT_PLAN_INVALID",
                    "Unknown feature dtype/shape",
                    field_path="data.samples.fields",
                ) from error
            if actual_bytes != sample["nbytes"]:
                raise PipelineError(
                    "INPUT_PLAN_INVALID",
                    "Feature bytes disagree with shape/dtype",
                    field_path="data.samples.nbytes",
                )
        sample.update(
            producer_replica=planned_producer(config, position),
            reader_ranks=list(planned_readers(config, position)),
            update_index=position // 4,
        )
    capacity = int(config["store"]["pool_bytes"] * config["store"]["utilization"])
    for start in range(0, len(samples), 4):
        if sum(s["nbytes"] for s in samples[start : start + 4]) > capacity:
            raise PipelineError(
                "UPDATE_CAPACITY_EXCEEDED",
                "Pool capacity cannot hold a complete update group",
                field_path="store.pool_bytes",
            )

    resources = {
        alias: {"gpus": 0, "allowed": set(), "cpu": {"node_agent": 1}}
        for alias in selected
    }
    for role in ("inference", "training"):
        for allocation in config[role]["nodes"]:
            alias = allocation["node"]
            node = selected[alias]
            count = (
                4 * config["inference"]["dp"]
                if role == "inference"
                else allocation["gpus"]
            )
            allowed = set(allocation.get("allowed_gpu_uuids", ()))
            if (
                allowed - {g["uuid"] for g in node["gpus"]}
                or allowed & resources[alias]["allowed"]
            ):
                raise PipelineError(
                    "GPU_IDENTITY_CONFLICT",
                    "Allowed GPU UUIDs are absent or overlap another role",
                    field_path=f"{role}.nodes",
                )
            resources[alias]["allowed"].update(allowed)
            resources[alias]["gpus"] += count
            if resources[alias]["gpus"] > min(len(node["gpus"]), node["gpu_available"]):
                raise PipelineError(
                    "INSUFFICIENT_GPU",
                    "Node GPU capacity cannot satisfy the complete layout",
                    field_path=f"{role}.nodes",
                    exit_code=4,
                )

    inference_aliases = [n["node"] for n in config["inference"]["nodes"]]
    core_alias = inference_aliases[0]
    resources[core_alias]["cpu"].update(
        frontend=1, native_cores=config["inference"]["dp"], gate=1
    )
    placement_groups, replicas = [], []
    for replica_id in range(config["inference"]["dp"]):
        worker_aliases = [alias for alias in inference_aliases for _ in range(4)]
        worker_nodes = tuple(selected[a]["node_id"] for a in worker_aliases)
        replica = InferenceReplica(
            replica_id,
            config["inference"]["tp"],
            worker_nodes,
            selected[core_alias]["node_id"],
            config["inference"]["tp"],
        )
        replicas.append(replica.to_dict())
        bundles = [
            {
                "node_id": selected[a]["node_id"],
                "resources": {"GPU": 1, f"node:{selected[a]['ip']}": 0.001},
            }
            for a in worker_aliases
        ]
        bundles.append(
            {
                "node_id": selected[core_alias]["node_id"],
                "resources": {"CPU": 1, f"node:{selected[core_alias]['ip']}": 0.001},
            }
        )
        placement_groups.append(
            {
                "id": f"inference-{replica_id}",
                "owner": "DeepSpec",
                "borrower": "vLLM",
                "bundles": bundles,
            }
        )
    training_nodes = []
    for allocation in config["training"]["nodes"]:
        alias, local_world = allocation["node"], allocation["gpus"]
        node = selected[alias]
        training_nodes.append((node["node_id"], local_world))
        resources[alias]["cpu"]["training_launcher"] = 2 * local_world
        placement_groups.append(
            {
                "id": f"training-{len(training_nodes) - 1}",
                "owner": "DeepSpec",
                "borrower": None,
                "bundles": [
                    {
                        "node_id": node["node_id"],
                        "resources": {
                            "CPU": 2 * local_world,
                            "GPU": local_world,
                            f"node:{node['ip']}": 0.001,
                        },
                    }
                ],
            }
        )
    ranks = training_rank_table(training_nodes, tp=4, dp=config["training"]["dp"])
    store_alias = config["store"]["node"]
    resources[store_alias]["cpu"].update(feature_buffer=1, verifier=1)
    if config["store"]["master"]["mode"] == "owned":
        resources[store_alias]["cpu"]["master"] = 1
    cpu_budgets, node_budgets = {}, {}
    for spec in config["nodes"]:
        alias = spec["alias"]
        node = selected[alias]
        components = resources[alias]["cpu"]
        total = sum(components.values())
        if total > min(spec["cpu_limit"], node["cpu_available"]):
            raise PipelineError(
                "INSUFFICIENT_CPU",
                f"CPU bill {total} exceeds node available/declared limit",
                field_path=f"nodes.{alias}.cpu_limit",
                exit_code=4,
            )
        cpu_budgets[node["node_id"]] = {
            "components": components,
            "total": total,
            "limit": spec["cpu_limit"],
        }
        try:
            node_budgets[node["node_id"]] = node_feature_budget(
                pool_bytes=config["store"]["pool_bytes"] if alias == store_alias else 0,
                max_sample_bytes=max(s["nbytes"] for s in samples),
                writers=config["inference"]["dp"] if alias == core_alias else 0,
                writer_inflight=config["inference"]["writer_inflight"],
                readers=sum(r["node_id"] == node["node_id"] for r in ranks),
                gas=counts["gas"],
                prefetch_depth=config["transport"]["prefetch_depth"],
                prefetch_bytes=config["transport"].get("prefetch_bytes"),
                snapshot=node["memory"],
                feature_cap=spec.get("feature_memory_cap_bytes"),
            )
        except ValueError as error:
            raise PipelineError(
                "INSUFFICIENT_MEMORY",
                str(error),
                node_id=node["node_id"],
                field_path=f"nodes.{alias}.memory",
                exit_code=4,
            ) from error
    return TopologyPlan.freeze(
        {
            "run_id": run_id,
            "layout": config["layout"],
            "config": config,
            "config_hash": content_hash(config),
            "input_plan_hash": inputs.get("native_plan_hash", content_hash(inputs)),
            "sample_plan_hash": content_hash(inputs),
            "input_plan": inputs,
            "samples": samples,
            "counts": counts,
            "nodes": selected,
            "replicas": replicas,
            "training_ranks": ranks,
            "placement_groups": placement_groups,
            "cpu_budgets": cpu_budgets,
            "node_budgets": node_budgets,
            "timeouts_seconds": config["timeouts_seconds"],
            "services": {
                "pool_node_id": selected[store_alias]["node_id"],
                "master": config["store"]["master"],
            },
        }
    )


def prepare_input(config, run):
    """Reuse native CPU preparation while preserving its exact persisted bytes."""
    import hashlib

    import torch

    from .run import prepare
    from .runtime import atomic_json

    output = Path(run.output_dir)
    legacy = {
        "schema_version": 2,
        "run_id": run.run_id,
        "namespace": run.namespace,
        "buffer_name": "features",
        "model_path": config["model_path"],
        "source_path": config["data"]["source_path"],
        "output_dir": run.output_dir,
        "context_length": config["data"]["context_length"],
        "epochs": config["data"]["epochs"],
        "steps": config["training"]["steps"],
        "samples_per_update": 4,
        "consumer_world_size": 4 * config["training"]["dp"],
        "consumer_dp": config["training"]["dp"],
        "consumer_nodes": len(config["training"]["nodes"]),
        "producer_dp": config["inference"]["dp"],
        "producer_batch_size": config["inference"]["batch_size"],
        "writer_inflight": config["inference"]["writer_inflight"],
        "window": config["transport"]["window"],
        "prefetch_depth": config["transport"]["prefetch_depth"],
        "prefetch_bytes": config["transport"].get("prefetch_bytes"),
        "pool_bytes": config["store"]["pool_bytes"],
        "pool_utilization": config["store"]["utilization"],
        "cluster_address": config["ray_address"],
        "verify_transfers": True,
        "receive_device": config["transport"]["receive_device"],
        "timeout_seconds": config["timeouts_seconds"]["transfer"],
        "events_path": str(output / "events.jsonl"),
        "store": {
            "protocol": config["transport"]["protocol"],
            "rdma_devices": config["transport"]["rdma_devices"],
        },
    }
    compatibility_path = output / "pipeline.json"
    atomic_json(compatibility_path, legacy)
    prepare(
        legacy,
        compatibility_path,
        timeout_seconds=config["timeouts_seconds"]["initialization"],
        validate_legacy_buffer=False,
    )
    raw = (output / "inputs/input-plan.json").read_bytes()
    inputs = json.loads(raw)
    inputs["native_plan_hash"] = hashlib.sha256(raw).hexdigest()
    # Read only safetensors headers. Four CPU copies of the largest full tensor
    # at eight bytes/element conservatively cover native DCP materialization,
    # hash conversion and optimizer dtype, plus metadata/interpreter overhead.
    # This budget is checked before reserving GPUs and again in the CPU actor.
    largest = 0
    for shard in sorted(Path(config["model_path"]).glob("*.safetensors")):
        with shard.open("rb") as stream:
            length = int.from_bytes(stream.read(8), "little")
            if not 0 < length <= 128 * 1024**2:
                raise ValueError("Invalid safetensors metadata length")
            header = json.loads(stream.read(length))
        largest = max(
            largest,
            *(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__"),
        )
    if largest <= 0:
        raise PipelineError(
            "VERIFIER_BUDGET_MISSING",
            "Cannot estimate checkpoint tensors from model metadata",
            exit_code=4,
        )
    inputs["verification_memory_bytes"] = 4 * 8 * largest + 1024**3
    inputs["batches"] = legacy["samples"]
    for sample in inputs["batches"]:
        batch = torch.load(sample["input_path"], map_location="cpu", weights_only=True)
        fields = {
            name: {"shape": list(t.shape), "dtype": str(t.dtype).removeprefix("torch.")}
            for name, t in batch.items()
        }
        length, teacher = sample["length"], legacy["teacher"]
        fields.update(
            seq_len={"shape": [], "dtype": "int64"},
            context_chunk_len={"shape": [], "dtype": "int64"},
            target_hidden_states={
                "shape": [
                    1,
                    length,
                    teacher["hidden_size"] * len(teacher["target_layer_ids"]),
                ],
                "dtype": "bfloat16",
            },
            target_last_hidden_states={
                "shape": [1, length, teacher["hidden_size"]],
                "dtype": "bfloat16",
            },
        )
        sample["fields"] = fields
    # inputs contains planning annotations; do not rewrite native input-plan.json.
    return inputs, legacy
