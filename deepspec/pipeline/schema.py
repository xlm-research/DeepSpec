"""Versioned transport settings with a compatibility upgrade path.

Existing preparation files predate the explicit transport section.  Readers
accept those files as schema version 1 and materialize the version 2 defaults
without changing the descriptor or ledger fields used by the native trainer.
"""

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from .store import CHUNK_BYTES

CURRENT_SCHEMA_VERSION = 2
TASK_SCHEMA_VERSION = 3


def canonical_json(value):
    """Only finite JSON data may cross the persistent/runtime boundary."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class TaskConfig:
    """An owned immutable value; access returns copies, never shared containers."""

    _json: str

    @classmethod
    def from_dict(cls, value):
        return cls(canonical_json(value))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def config_hash(self):
        return hashlib.sha256(self._json.encode()).hexdigest()


def upgrade_task_config(value):
    """Translate explicit legacy settings before applying the v3 schema.

    This is a new task declaration, not a resume of a legacy runtime ledger.
    Historical split-DP runs need explicit node selectors: DP alone cannot
    distinguish the old shared-node layout from the later separated layout.
    """
    from .runtime import PipelineError

    version = value.get("schema_version", 1)
    if version == TASK_SCHEMA_VERSION:
        return normalize_task_config(value)
    if type(version) is not int or version not in (1, 2):
        raise PipelineError(
            "INVALID_CONFIG",
            "Expected schema version 1, 2 or 3",
            field_path="schema_version",
        )

    def read(path):
        item = value
        for key in path.split("."):
            if not isinstance(item, dict) or key not in item:
                return None
            item = item[key]
        return item

    def choose(*paths, default=None):
        explicit = [(path, read(path)) for path in paths if read(path) is not None]
        if explicit and any(item != explicit[0][1] for _, item in explicit[1:]):
            raise PipelineError(
                "LEGACY_CONFIG_CONFLICT",
                "Conflicting values for " + " and ".join(p for p, _ in explicit),
                field_path=paths[-1],
            )
        return copy.deepcopy(explicit[0][1] if explicit else default)

    def require_equal(actual, expected, path):
        if actual != expected:
            raise PipelineError(
                "LEGACY_CONFIG_CONFLICT",
                f"{path} disagrees with the declared role topology",
                field_path=path,
            )

    dp = choose("consumer_dp", "training.dp", default=1)
    pdp = choose("producer_dp", "inference.dp", default=1)
    if (
        type(dp) is not int
        or dp not in (1, 2)
        or type(pdp) is not int
        or pdp not in (1, 2)
    ):
        raise PipelineError(
            "UNSUPPORTED_TOPOLOGY",
            "Legacy tasks require DP1 or DP2",
            field_path="consumer_dp",
        )
    require_equal(
        choose("consumer_world_size", default=4 * dp),
        4 * dp,
        "consumer_world_size/training.dp",
    )
    count = value.get("consumer_nodes")
    if count is None:
        if dp == 1 or value.get("role_separation") is True:
            count = 1
        else:
            raise PipelineError(
                "LEGACY_NODE_SELECTION_REQUIRED",
                "Specify consumer_nodes and selectors for historical split-DP topology",
                field_path="consumer_nodes",
            )
    if type(count) is not int or count not in (1, 2) or (count == 2 and dp != 2):
        raise PipelineError(
            "UNSUPPORTED_TOPOLOGY",
            "Invalid consumer_nodes for training DP",
            field_path="consumer_nodes",
        )
    producer = value.get("producer_node") or value.get("producer_node_id")
    consumers = value.get("consumer_node_ids")
    if consumers is None:
        consumer = value.get("consumer_node") or value.get("consumer_node_id")
        if not value.get("cluster_address") and not value.get("role_separation"):
            producer = producer or read("store.host")
            consumer = consumer or producer
        consumers = [consumer] if consumer else []
    if not producer or len(consumers) != count or any(not n for n in consumers):
        raise PipelineError(
            "LEGACY_NODE_SELECTION_REQUIRED",
            "Explicit producer_node and consumer node selectors are required",
            field_path="consumer_nodes",
        )
    selectors = list(dict.fromkeys([producer, *consumers]))
    names = {}
    nodes = []
    for index, selector in enumerate(selectors):
        try:
            ipaddress.ip_address(selector)
            key = "ip"
        except ValueError:
            key = "node_id"
        declarations = [
            n for n in value.get("nodes", []) if n.get("selector") == {key: selector}
        ]
        if "nodes" in value and len(declarations) != 1:
            raise PipelineError(
                "LEGACY_CONFIG_CONFLICT",
                "nodes must match each legacy role selector exactly once",
                field_path="nodes",
            )
        node = (
            copy.deepcopy(declarations[0])
            if declarations
            else {
                "alias": f"node-{index}",
                "selector": {key: selector},
                "cpu_limit": value.get("cpu_limit", 24),
            }
        )
        names[selector] = node["alias"]
        if "cpu_limit" in value:
            require_equal(
                node["cpu_limit"], value["cpu_limit"], "cpu_limit/nodes.cpu_limit"
            )
        cap = choose("feature_memory_budget", "feature_memory_cap_bytes")
        if cap is not None:
            require_equal(
                node.get("feature_memory_cap_bytes", cap),
                cap,
                "feature_memory_cap_bytes/nodes.feature_memory_cap_bytes",
            )
            node["feature_memory_cap_bytes"] = cap
        nodes.append(node)
    if "nodes" in value:
        require_equal(len(value["nodes"]), len(nodes), "nodes")
    layout = "M0" if len(selectors) == 1 else ("M3" if count == 2 else "M1")
    if "layout" in value:
        require_equal(value["layout"], layout, "layout")
    inference_nodes = [{"node": names[producer], "gpus": 4 * pdp}]
    training_nodes = [{"node": names[n], "gpus": 4 * dp // count} for n in consumers]
    for role, expected in (
        ("inference", inference_nodes),
        ("training", training_nodes),
    ):
        if read(f"{role}.nodes") is not None:
            require_equal(read(f"{role}.nodes"), expected, f"{role}.nodes")
    store = value.get("store", {})
    if "node" in store:
        require_equal(store["node"], names[consumers[0]], "store.node")
    master = store.get("master", {"mode": "owned"})
    if isinstance(master, str):
        master = {"mode": "owned", "endpoint": master}
    transport = {
        field: choose(*paths, default=default)
        for field, paths, default in (
            ("protocol", ("store.protocol", "transport.protocol"), "tcp"),
            ("rdma_devices", ("store.rdma_devices", "transport.rdma_devices"), ""),
            ("receive_device", ("receive_device", "transport.receive_device"), "cpu"),
            ("window", ("window", "transport.window"), 8),
            ("prefetch_depth", ("prefetch_depth", "transport.prefetch_depth"), 2),
            ("prefetch_bytes", ("prefetch_bytes", "transport.prefetch_bytes"), None),
        )
    }
    if transport["prefetch_bytes"] is None:
        del transport["prefetch_bytes"]
    verify = choose(
        "store.verify_mode",
        "transport.verify_mode",
        default="full" if value.get("verify_transfers", True) else "none",
    )
    if "verify_transfers" in value:
        require_equal(
            verify,
            "full" if value["verify_transfers"] else "none",
            "verify_transfers/transport.verify_mode",
        )
    transport["verify_mode"] = verify
    policy = {
        "allocation": 120,
        "initialization": 1800,
        "run": 7200,
        "transfer": 1800,
        "collective": 600,
        "cleanup": 120,
        "lease": 30,
        "heartbeat": 5,
        "budget_snapshot": 5,
    }
    for field in ("initialization", "run", "transfer"):
        policy[field] = choose(
            "timeout_seconds", f"timeouts_seconds.{field}", default=policy[field]
        )
    policy.update(
        {
            k: v
            for k, v in value.get("timeouts_seconds", {}).items()
            if k not in ("initialization", "run", "transfer")
        }
    )
    result = {
        "schema_version": 3,
        "layout": layout,
        "ray_address": choose("cluster_address", "ray_address", default="auto")
        or "auto",
        "gpu_sharing": choose("gpu_sharing", default="exclusive"),
        "model_path": value.get("model_path"),
        "output_dir": value.get("output_dir"),
        "nodes": nodes,
        "inference": {
            "tp": choose("producer_tp", "inference.tp", default=4),
            "dp": pdp,
            "nodes": inference_nodes,
            "batch_size": choose(
                "producer_batch_size", "inference.batch_size", default=1
            ),
            "writer_inflight": choose(
                "writer_inflight",
                "store.async_put_pool_size",
                "transport.async_put_pool_size",
                "inference.writer_inflight",
                default=1,
            ),
        },
        "training": {
            "tp": 4,
            "dp": dp,
            "dp_mode": "shard",
            "cp": 1,
            "pp": 1,
            "nodes": training_nodes,
            "global_batch_size": choose(
                "samples_per_update", "training.global_batch_size", default=4
            ),
            "steps": choose("steps", "training.steps", default=3),
        },
        "data": {
            "source_path": choose("source_path", "data.source_path"),
            "context_length": choose(
                "context_length", "data.context_length", default=4096
            ),
            "epochs": choose("epochs", "data.epochs", default=1),
        },
        "store": {
            "node": names[consumers[0]],
            "master": master,
            "pool_bytes": choose("pool_bytes", "store.pool_bytes", default=4 * 1024**3),
            "utilization": choose(
                "pool_utilization", "store.utilization", default=0.75
            ),
        },
        "transport": transport,
        "timeouts_seconds": policy,
    }
    # New grouped fields remain strict even within a v1/v2 compatibility file.
    for section in ("inference", "training", "data"):
        for field, item in value.get(section, {}).items():
            if field not in result[section]:
                raise PipelineError(
                    "INVALID_CONFIG",
                    "Unknown grouped field",
                    field_path=f"{section}.{field}",
                )
            require_equal(item, result[section][field], f"{section}.{field}")
    for field in value.get("transport", {}):
        if field not in transport and field not in (
            "async_put_pool_size",
            "chunk_bytes",
            "wait_for_visibility",
        ):
            raise PipelineError(
                "INVALID_CONFIG",
                "Unknown grouped field",
                field_path=f"transport.{field}",
            )
    for field, expected in (
        ("retain_for_peak", False),
        ("transport.chunk_bytes", CHUNK_BYTES),
        ("transport.wait_for_visibility", True),
        ("store.wait_for_visibility", True),
    ):
        if read(field) is not None and read(field) != expected:
            raise PipelineError(
                "UNSUPPORTED_LEGACY_OPTION",
                f"{field} cannot be represented by the v3 task contract",
                field_path=field,
            )
    return normalize_task_config(result)


def normalize_task_config(value):
    """Validate v3 before any runtime work and materialize hashed defaults."""
    from jsonschema import Draft202012Validator

    from .runtime import PipelineError

    def reject(message, path, code="INVALID_CONFIG"):
        raise PipelineError(code, message, field_path=path)

    def check_scalars(item, path="$"):
        if isinstance(item, dict):
            for key, child in item.items():
                check_scalars(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                check_scalars(child, f"{path}[{index}]")
        elif isinstance(item, str) and "${" in item:
            reject(
                "Replace template placeholders explicitly before running preview", path
            )
        elif isinstance(item, bool) or (
            isinstance(item, float) and not math.isfinite(item)
        ):
            reject("Expected finite numeric values, not booleans or NaN/infinity", path)

    check_scalars(value)
    config = copy.deepcopy(value)
    schema = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "specs/001-unify-ray-topology/contracts/task-config.schema.json"
        ).read_text()
    )
    errors = sorted(
        Draft202012Validator(schema).iter_errors(config),
        key=lambda e: str(list(e.absolute_path)),
    )
    if errors:
        error = errors[0]
        path = ".".join(map(str, error.absolute_path)) or "$"
        reject(error.message, path)
    config.setdefault("gpu_sharing", "exclusive")
    config["transport"].setdefault("rdma_devices", "")
    config["timeouts_seconds"].setdefault("budget_snapshot", 5)
    if config["timeouts_seconds"]["heartbeat"] >= config["timeouts_seconds"]["lease"]:
        reject(
            "Heartbeat interval must be smaller than lease duration",
            "timeouts_seconds.heartbeat",
        )
    names = [n["alias"] for n in config["nodes"]]
    selectors = [canonical_json(n["selector"]) for n in config["nodes"]]
    if len(set(names)) != len(names) or len(set(selectors)) != len(selectors):
        reject("Node aliases and selectors must be unique", "nodes")
    required_nodes = {"M0": 1, "M1": 2, "M2": 3, "M3": 3}[config["layout"]]
    if len(names) != required_nodes:
        reject(
            f"{config['layout']} requires {required_nodes} nodes",
            "nodes",
            "UNSUPPORTED_TOPOLOGY",
        )
    roles = {}
    for role in ("inference", "training"):
        allocations = config[role]["nodes"]
        roles[role] = [a["node"] for a in allocations]
        if len(set(roles[role])) != len(allocations) or any(
            n not in names for n in roles[role]
        ):
            reject("Role nodes must be unique declared aliases", f"{role}.nodes")
        needed = (
            4 * config[role]["dp"] // (len(allocations) if role == "training" else 1)
        )
        for i, allocation in enumerate(allocations):
            quota = allocation["gpus"]
            if (
                role == "inference" and config["layout"] == "M2" and quota < needed
            ) or (
                not (role == "inference" and config["layout"] == "M2")
                and quota != needed
            ):
                reject(
                    f"Insufficient or unsupported per-node capacity: need {needed} GPUs",
                    f"{role}.nodes.{i}.gpus",
                    "UNSUPPORTED_TOPOLOGY",
                )
            if (
                "allowed_gpu_uuids" in allocation
                and len(allocation["allowed_gpu_uuids"]) < needed
            ):
                reject(
                    "Allowed GPU UUIDs cannot satisfy role capacity",
                    f"{role}.nodes.{i}.allowed_gpu_uuids",
                )
    overlap = set(roles["inference"]) & set(roles["training"])
    if config["layout"] != "M0" and overlap:
        reject("Inference and training must occupy separate nodes", "training.nodes")
    if config["store"]["node"] != roles["training"][0]:
        reject("Pool/master must be on the first training node", "store.node")
    endpoint = config["store"]["master"].get("endpoint")
    if endpoint:
        try:
            parsed = urlsplit("//" + endpoint)
            host, port = parsed.hostname, parsed.port
            if not host or port is None or not 1 <= port <= 65535 or parsed.path:
                raise ValueError("Expected host:port")
            try:
                address = ipaddress.ip_address(host)
            except ValueError:
                address = None
            if host.lower() == "localhost" or (
                address and (address.is_loopback or address.is_unspecified)
            ):
                raise ValueError("Store endpoint must be routable across participants")
        except ValueError as error:
            reject(str(error), "store.master.endpoint")
    return TaskConfig.from_dict(config)


def normalize_pipeline_config(config):
    """Upgrade a pipeline dictionary in place and return it.

    Keeping the old top-level keys is intentional: old launchers and manifests
    can be resumed while new components consume the grouped ``transport``
    settings.  A future schema can reject incompatible values here before any
    Ray actor or Mooncake client is started.
    """

    version = int(config.get("schema_version", 1))
    if version < 1 or version > CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported DeepSpec pipeline schema {version}; "
            f"expected 1..{CURRENT_SCHEMA_VERSION}"
        )
    transport = config.setdefault("transport", {})
    if not isinstance(transport, dict):
        raise TypeError("pipeline transport settings must be an object")
    transport.setdefault("chunk_bytes", CHUNK_BYTES)
    transport.setdefault(
        "verify_mode", "full" if config.get("verify_transfers", True) else "none"
    )
    transport.setdefault("prefetch_depth", int(config.get("prefetch_depth", 2)))
    transport.setdefault("prefetch_bytes", config.get("prefetch_bytes"))
    transport.setdefault("async_put_pool_size", int(config.get("writer_inflight") or 1))
    transport.setdefault("wait_for_visibility", True)
    transport["chunk_bytes"] = int(transport["chunk_bytes"])
    transport["prefetch_depth"] = int(transport["prefetch_depth"])
    if transport["prefetch_bytes"] is not None:
        transport["prefetch_bytes"] = int(transport["prefetch_bytes"])
    transport["async_put_pool_size"] = int(transport["async_put_pool_size"])
    if transport["prefetch_depth"] < 1:
        raise ValueError("transport.prefetch_depth must be positive")
    if transport["verify_mode"] not in ("full", "none"):
        raise ValueError("transport.verify_mode must be 'full' or 'none'")
    if transport["prefetch_bytes"] is not None and transport["prefetch_bytes"] <= 0:
        raise ValueError("transport.prefetch_bytes must be positive")
    if transport["chunk_bytes"] <= 0:
        raise ValueError("transport.chunk_bytes must be positive")
    store = config.setdefault("store", {})
    if not isinstance(store, dict):
        raise TypeError("pipeline store settings must be an object")
    store.setdefault("async_put_pool_size", transport["async_put_pool_size"])
    store.setdefault("verify_mode", transport["verify_mode"])
    store.setdefault("wait_for_visibility", transport["wait_for_visibility"])
    config["prefetch_depth"] = transport["prefetch_depth"]
    if transport["prefetch_bytes"] is not None:
        config["prefetch_bytes"] = transport["prefetch_bytes"]
    config["schema_version"] = CURRENT_SCHEMA_VERSION
    return config
