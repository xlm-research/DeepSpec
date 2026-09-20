"""Single-node 4+4 real-model pilot; run with the user's existing Python env."""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

from .memory import GIB, feature_budget
from .topology import (
    consumer_dp,
    consumer_nodes,
    producer_dp,
    sample_producer,
    sample_readers,
)

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def environment(model_path, *, inference_tp=4):
    return {
        "PYTHONPATH": ":".join(map(str, (ROOT, ROOT / "torchtitan", ROOT / "vllm"))),
        "TARGET_MODEL_PATH": model_path,
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
        "VLLM_RAY_BUNDLE_INDICES": ",".join(map(str, range(inference_tp))),
    }


def prepare(config, config_path, *, timeout_seconds=None, validate_legacy_buffer=True):
    import torch

    from deepspec.trainer.qwen3_8_vllm import teacher_identity

    from .schema import normalize_pipeline_config

    normalize_pipeline_config(config)

    output = Path(config["output_dir"])
    request = {
        "recipe_args": [
            "--module",
            "deepspec.pipeline.recipe",
            "--config",
            "qwen38_preparation",
        ],
        "workers": config["consumer_world_size"],
        "output_dir": str(output / "inputs"),
        "run_id": config["run_id"],
    }
    write_json(output / "preparation-request.json", request)
    child_env = dict(
        os.environ,
        **environment(config["model_path"]),
        DEEPSPEC_PIPELINE_CONFIG=str(config_path),
        CUDA_VISIBLE_DEVICES="",
    )
    with (output / "preparation.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torchtitan.models.dspark_draft.preparation",
                str(output / "preparation-request.json"),
            ],
            env=child_env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=timeout_seconds,
        )
    plan_path = output / "inputs" / "input-plan.json"
    plan = json.loads(plan_path.read_text())
    if (
        plan["data_parallel_size"] != consumer_dp(config)
        or plan["global_batch_size"] != config["samples_per_update"]
        or plan["gradient_accumulation_steps"]
        != config["samples_per_update"] // consumer_dp(config)
    ):
        raise ValueError("Native input preparation changed the consumer batch layout")
    teacher = teacher_identity(
        config["model_path"], plan["producer_requirements"]["target_layer_ids"]
    )
    config.update(
        teacher=teacher,
        plan_path=str(plan_path),
        samples=plan["batches"],
        manifest_path=str(output / "manifest.json"),
    )
    for sample in config["samples"]:
        batch = torch.load(sample["input_path"], map_location="cpu", weights_only=True)
        sample["nbytes"] = (
            sum(t.numel() * t.element_size() for t in batch.values())
            + 16
            + sample["length"]
            * teacher["hidden_size"]
            * (len(teacher["target_layer_ids"]) + 1)
            * 2
        )
    write_json(
        config["manifest_path"],
        {"batches": [{"id": x["id"]} for x in config["samples"]]},
    )
    largest = max(x["nbytes"] for x in config["samples"])
    config["max_sample_nbytes"] = largest
    prefetch_depth = int(config.get("prefetch_depth", 2))
    if prefetch_depth < 1:
        raise ValueError("Prefetch depth must be positive")
    config["prefetch_depth"] = prefetch_depth
    if config.get("prefetch_bytes") is None:
        config["prefetch_bytes"] = largest * prefetch_depth
    config["transport"]["prefetch_depth"] = prefetch_depth
    config["transport"]["prefetch_bytes"] = config["prefetch_bytes"]
    config["store"].setdefault(
        "async_put_pool_size", int(config.get("writer_inflight") or 1)
    )
    config["store"].setdefault("wait_for_visibility", True)
    if not config.get("cluster_address"):
        config.update(
            feature_budget(
                config["pool_bytes"],
                largest,
                config["window"],
                config["consumer_world_size"],
                config["samples_per_update"],
                writer_inflight=config.get("writer_inflight"),
            )
        )
    config["capacity_bytes"] = int(
        config["pool_bytes"] * config.get("pool_utilization", 0.75)
    )
    config["retain_until_bytes"] = 0
    if config.get("retain_for_peak"):
        for sample in config["samples"][: config["window"]]:
            target = config["retain_until_bytes"] + sample["nbytes"]
            if target > config["capacity_bytes"]:
                break
            config["retain_until_bytes"] = target
        if config["retain_until_bytes"] < config["pool_bytes"] * 0.95:
            raise ValueError(
                "Peak mode needs enough samples/window to fill at least 95% of the pool"
            )
    from .buffer import BufferLedger

    if validate_legacy_buffer:
        _validate_prepared_legacy_buffer(config, BufferLedger)
    write_json(config_path, config)


def _validate_prepared_legacy_buffer(config, ledger_type):
    ledger_type(
        config["samples"],
        capacity=config["capacity_bytes"],
        window=config["window"],
        readers=range(config["consumer_world_size"]),
        samples_per_update=config["samples_per_update"],
        readers_by_position=[
            sample_readers(config, sample["position"]) for sample in config["samples"]
        ],
        producer_dp=producer_dp(config),
    )


def require_idle_gpus():
    busy = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    gpus = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    if busy or len(gpus.strip().splitlines()) != 8:
        raise RuntimeError(
            f"The 4+4 pilot requires eight idle GPUs; current processes: {busy}"
        )
    return gpus


def check_gpu_ownership(output, run_id, known_owned):
    from deepspec.orchestration.process import descendants

    from .cluster import gpu_processes

    # An exiting worker can be reparented to the outer subreaper while NVML
    # still reports it. Reuse the run marker + PID/start-time ownership cache.
    processes = gpu_processes(run_id, known_owned)
    owned_descendants = descendants(os.getpid()) | {os.getpid()}
    for process in processes:
        # Ray can start CUDA helpers before an actor's runtime env is applied.
        # Retain the original descendant check, then cache the exact identity.
        if process["pid"] in owned_descendants:
            known_owned.add((process["pid"], process["process_start_ticks"]))
            process["owned"] = True
        process["memory_mib"] = process.pop("used_mib")
        if not process["owned"]:
            try:
                process["command"] = (
                    (Path("/proc") / str(process["pid"]) / "cmdline")
                    .read_bytes()
                    .replace(b"\0", b" ")
                    .decode(errors="replace")
                )
            except OSError:
                process["command"] = "<process exited during inspection>"
    with (output / "gpu-status.jsonl").open("a") as log:
        log.write(json.dumps({"time": time.time(), "processes": processes}) + "\n")
    foreign = [row for row in processes if not row["owned"]]
    if foreign:
        raise RuntimeError(f"Another job started using the pilot GPUs: {foreign}")


def merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Event interval ends before it starts")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def summarize_events(events, config):
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    expected_reads = Counter(
        (sample["position"], rank)
        for sample in config["samples"]
        for rank in sample_readers(config, sample["position"])
    )
    for event in ("claimed", "received", "gpu_ready", "compute_start", "compute_end"):
        actual = Counter(
            (row["position"], row["reader"]) for row in rows if row["event"] == event
        )
        if actual != expected_reads:
            raise RuntimeError(f"Incomplete or duplicate rank events: {event}")
    expected_steps = Counter(
        (step, rank)
        for step in range(1, config["steps"] + 1)
        for rank in range(config["consumer_world_size"])
    )
    actual_steps = Counter(
        (row["step"], row["reader"])
        for row in rows
        if row["event"] == "optimizer_update_complete"
    )
    if actual_steps != expected_steps:
        raise RuntimeError("Ranks did not all complete the planned optimizer updates")
    if consumer_dp(config) > 1:
        gas = config["samples_per_update"] // consumer_dp(config)
        for row in rows:
            if row["event"] == "optimizer_update_complete" and (
                row.get("next_global_microbatch") != row["step"] * gas
                or row.get("next_global_sample")
                != row["step"] * config["samples_per_update"]
            ):
                raise RuntimeError(
                    "DP checkpoint cursor differs from completed samples"
                )
        ranks = [r for r in rows if r["event"] == "consumer_rank_initialized"]
        if Counter(r["reader"] for r in ranks) != Counter(
            range(config["consumer_world_size"])
        ) or any(
            r["dp_rank"] != r["reader"] // 4
            or r["tp_rank"] != r["reader"] % 4
            or r["world_size"] != config["consumer_world_size"]
            or r["gradient_accumulation_steps"] != gas
            or r["hostname"] != config["consumer_hostnames"][r["reader"] // 4]
            for r in ranks
        ):
            raise RuntimeError("Native ranks did not form the assigned DP/TP topology")
    gradients = Counter(
        row["reader"] for row in rows if row["event"] == "context_gradient_verified"
    )
    if gradients != Counter(range(config["consumer_world_size"])):
        raise RuntimeError("DSpark context gradients were not verified on every rank")
    producer_ids = {
        (r["node_id"], gpu)
        for r in rows
        if r["event"] == "producer_worker"
        for gpu in r["ray_gpu_ids"]
    }
    consumer_ids = {
        (r["node_id"], gpu)
        for r in rows
        if r["event"] == "consumer_launcher"
        for gpu in r["ray_gpu_ids"]
    }
    if (
        len(producer_ids) != 4 * producer_dp(config)
        or len(consumer_ids) != config["consumer_world_size"]
        or producer_ids & consumer_ids
    ):
        raise RuntimeError("Observed producer and consumer GPU assignments are invalid")
    if config.get("role_separation") and (
        {node for node, _ in producer_ids} & {node for node, _ in consumer_ids}
    ):
        raise RuntimeError("Producer and consumer must occupy separate physical nodes")
    if config.get("cluster_address") and (
        Counter(node for node, _ in producer_ids)
        != Counter(
            node
            for node in config.get("producer_node_ids", [config["producer_node_id"]])
            for _ in range(4)
        )
        or Counter(node for node, _ in consumer_ids)
        != Counter(
            {
                node: config["consumer_world_size"] // consumer_nodes(config)
                for node in config.get(
                    "consumer_node_ids", [config["consumer_node_id"]]
                )
            }
        )
    ):
        raise RuntimeError("Model workers did not use their assigned nodes")
    if producer_dp(config) > 1:
        workers = [r for r in rows if r["event"] == "producer_worker"]
        if Counter(
            (r.get("producer_rank"), r.get("tp_rank")) for r in workers
        ) != Counter(
            (dp, tp) for dp in range(producer_dp(config)) for tp in range(4)
        ) or any(
            len(r["ray_gpu_ids"]) != 1
            or r["node_id"] != config["producer_node_ids"][r["producer_rank"]]
            for r in workers
        ):
            raise RuntimeError(
                "Native producer workers did not form the assigned DP/TP topology"
            )
        expected_positions = Counter(s["position"] for s in config["samples"])
        for event in (
            "inference_start",
            "inference_end",
            "write_started",
            "ready",
            "write_complete",
        ):
            records = [r for r in rows if r["event"] == event]
            if Counter(r["position"] for r in records) != expected_positions or any(
                r.get("producer_rank") != sample_producer(config, r["position"])
                for r in records
            ):
                raise RuntimeError(
                    f"Incomplete, duplicate or wrongly routed production: {event}"
                )
        if [r["position"] for r in rows if r["event"] == "reserved"] != list(
            expected_positions
        ):
            raise RuntimeError("Production admission did not follow input-plan order")
        active = set()
        for row in rows:
            if row["event"] == "inference_start":
                if row["producer_rank"] in active:
                    raise RuntimeError(
                        "Concurrent generation within one producer DP group"
                    )
                active.add(row["producer_rank"])
            elif row["event"] == "inference_end":
                active.remove(row["producer_rank"])
    starts = {
        r["position"]: r["monotonic"] for r in rows if r["event"] == "inference_start"
    }
    intervals = merge_intervals(
        [
            (starts[r["position"]], r["monotonic"])
            for r in rows
            if r["event"] == "inference_end"
        ]
    )
    compute = {
        r["position"]: r["monotonic"]
        for r in rows
        if r["event"] == "compute_start" and r["reader"] == 0
    }
    overlap = 0.0
    transfer_overlap = 0.0
    transfer_starts = {
        row["position"]: row["monotonic"]
        for row in rows
        if row["event"] == "transfer_start" and row["reader"] == 0
    }
    transfers = [
        (transfer_starts[row["position"]], row["monotonic"])
        for row in rows
        if row["event"] == "transfer_end" and row["reader"] == 0
    ]
    for row in rows:
        if row["event"] == "compute_end" and row["reader"] == 0:
            for start, end in intervals:
                overlap += max(
                    0, min(end, row["monotonic"]) - max(start, compute[row["position"]])
                )
            for start, end in transfers:
                transfer_overlap += max(
                    0, min(end, row["monotonic"]) - max(start, compute[row["position"]])
                )
    return {
        "producer_gpu_ids": sorted(gpu for _, gpu in producer_ids),
        "consumer_gpu_ids": sorted(gpu for _, gpu in consumer_ids),
        "producer_gpu_assignments": sorted(producer_ids),
        "consumer_gpu_assignments": sorted(consumer_ids),
        "inference_compute_overlap_seconds": overlap,
        "transfer_compute_overlap_seconds": transfer_overlap,
        "backpressure_events": sum(r["event"] == "backpressure" for r in rows),
        "verified_rank_receives": sum(r["event"] == "received" for r in rows),
    }


def launch(config, config_path):
    """Compatibility API using the same frozen planning/controller entry."""
    from .legacy import run_config

    return run_config(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="/mnt/afs_agents/hongjiawei/share_models/Qwen/Qwen3.8-27B"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-length", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--pool-gib", type=int, default=4)
    parser.add_argument("--producer-batch-size", type=int, default=1)
    parser.add_argument("--writer-inflight", type=int)
    parser.add_argument("--prefetch-depth", type=int, default=2)
    parser.add_argument("--prefetch-bytes", type=int)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--pool-utilization", type=float, default=0.75)
    parser.add_argument("--retain-for-peak", action="store_true")
    parser.add_argument("--protocol", choices=("tcp", "rdma"), default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--receive-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--ray-address", default="", help="Existing two-node Ray Head IP:port"
    )
    parser.add_argument("--producer-node", default="", help="Ray node IP or node ID")
    parser.add_argument("--consumer-node", default="", help="Ray node IP or node ID")
    parser.add_argument(
        "--producer-dp",
        type=int,
        choices=(1, 2),
        default=1,
        help="Native inference DP, independent of consumer DP",
    )
    parser.add_argument(
        "--consumer-dp",
        type=int,
        choices=(1, 2),
        default=1,
        help="DP2 uses eight consumer GPUs on the consumer node",
    )
    parser.add_argument(
        "--transport-only",
        action="store_true",
        help="Check cross-node Store without loading GPU models",
    )
    args = parser.parse_args(argv)
    if (
        min(args.producer_batch_size, args.epochs, args.prefetch_depth) < 1
        or (args.writer_inflight is not None and args.writer_inflight < 1)
        or (args.prefetch_bytes is not None and args.prefetch_bytes < 1)
    ):
        parser.error(
            "Batch size, epochs, prefetch depth and byte limits must be positive"
        )
    if not 0 < args.pool_utilization <= 0.99:
        parser.error("Pool utilization must be in (0, 0.99]")
    if args.retain_for_peak and args.pool_utilization < 0.95:
        parser.error("Peak mode requires --pool-utilization >= 0.95")
    if args.ray_address and (not args.producer_node or not args.consumer_node):
        parser.error("--ray-address requires --producer-node and --consumer-node")
    if not args.ray_address and (
        args.producer_node
        or args.consumer_node
        or args.consumer_dp != 1
        or args.producer_dp != 1
    ):
        parser.error("Separate role nodes require --ray-address and explicit selectors")
    output = Path(args.output).resolve()
    run_id = f"dspark-{uuid.uuid4().hex[:12]}"
    host = socket.gethostbyname(socket.gethostname())
    config = {
        "run_id": run_id,
        "namespace": run_id,
        "buffer_name": "features",
        "model_path": str(Path(args.model).resolve()),
        "source_path": str(Path(args.source).resolve()),
        "output_dir": str(output),
        "context_length": args.context_length,
        "steps": args.steps,
        "epochs": args.epochs,
        "producer_batch_size": args.producer_batch_size,
        "writer_inflight": args.writer_inflight,
        "prefetch_depth": args.prefetch_depth,
        "prefetch_bytes": args.prefetch_bytes,
        "pool_utilization": args.pool_utilization,
        "retain_for_peak": args.retain_for_peak,
        "samples_per_update": 4,
        "consumer_world_size": 4 * args.consumer_dp,
        "consumer_dp": args.consumer_dp,
        "consumer_nodes": 1,
        "role_separation": bool(args.ray_address),
        "producer_dp": args.producer_dp,
        "window": args.window,
        "pool_bytes": args.pool_gib * GIB,
        "timeout_seconds": args.timeout_seconds,
        "receive_device": args.receive_device,
        "verify_transfers": True,
        "events_path": str(output / "events.jsonl"),
        "store": {
            "host": host,
            "master": {"mode": "owned"},
            "protocol": args.protocol,
            "rdma_devices": args.rdma_devices,
        },
        "cluster_address": args.ray_address,
        "producer_node": args.producer_node,
        "consumer_node": args.consumer_node,
        "transport_only": args.transport_only,
    }
    from .legacy import run_config
    from .runtime import PipelineError

    try:
        result = run_config(
            config, prepare_only=args.prepare_only, transport_only=args.transport_only
        )
    except PipelineError as error:
        print(json.dumps(error.to_dict()), file=sys.stderr)
        return error.exit_code
    print(json.dumps(result))
    return (
        0
        if args.prepare_only
        or result.get("state") == "succeeded"
        or result.get("status") == "passed"
        else 130
        if result.get("state") == "cancelled"
        else 3
    )


if __name__ == "__main__":
    raise SystemExit(main())
