"""Single-node 4+4 real-model pilot; run with the user's existing Python env."""

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

from .memory import GIB, feature_budget
from .store import free_port

ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def environment(model_path):
    return {
        "PYTHONPATH": ":".join(map(str, (ROOT, ROOT / "torchtitan", ROOT / "vllm"))),
        "TARGET_MODEL_PATH": model_path,
        "OMP_NUM_THREADS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "VLLM_USE_V2_MODEL_RUNNER": "0",
        "VLLM_USE_RAY_V2_EXECUTOR_BACKEND": "1",
        "VLLM_RAY_BUNDLE_INDICES": "0,1,2,3",
    }


def prepare(config, config_path):
    import torch

    from deepspec.trainer.qwen3_8_vllm import teacher_identity

    output = Path(config["output_dir"])
    request = {
        "recipe_args": [
            "--module",
            "deepspec.pipeline.recipe",
            "--config",
            "qwen38_preparation",
        ],
        "workers": 4,
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
        )
    plan_path = output / "inputs" / "input-plan.json"
    plan = json.loads(plan_path.read_text())
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
    config.update(
        feature_budget(
            config["pool_bytes"],
            largest,
            config["window"],
            config["consumer_world_size"],
            config["samples_per_update"],
        )
    )
    config["capacity_bytes"] = int(config["pool_bytes"] * 0.75)
    from .buffer import BufferLedger

    BufferLedger(
        config["samples"],
        capacity=config["capacity_bytes"],
        window=config["window"],
        readers=range(4),
        samples_per_update=config["samples_per_update"],
    )
    write_json(config_path, config)


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


def check_gpu_ownership(output):
    from deepspec.orchestration.process import descendants

    rows = (
        subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
        .strip()
        .splitlines()
    )
    owned = descendants(os.getpid()) | {os.getpid()}
    processes = []
    for row in rows:
        pid, gpu_uuid, memory = (value.strip() for value in row.split(","))
        processes.append(
            {
                "pid": int(pid),
                "gpu_uuid": gpu_uuid,
                "memory_mib": memory,
                "owned": int(pid) in owned,
            }
        )
    with (output / "gpu-status.jsonl").open("a") as log:
        log.write(json.dumps({"time": time.time(), "processes": processes}) + "\n")
    foreign = [
        row
        for row in processes
        if not row["owned"] and (Path("/proc") / str(row["pid"])).exists()
    ]
    if foreign:
        raise RuntimeError(f"Another job started using the pilot GPUs: {foreign}")


def summarize_events(events, config):
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    expected_reads = Counter(
        (sample["position"], rank)
        for sample in config["samples"]
        for rank in range(config["consumer_world_size"])
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
    gradients = Counter(
        row["reader"] for row in rows if row["event"] == "context_gradient_verified"
    )
    if gradients != Counter(range(config["consumer_world_size"])):
        raise RuntimeError("DSpark context gradients were not verified on every rank")
    producer_ids = {
        gpu for r in rows if r["event"] == "producer_worker" for gpu in r["ray_gpu_ids"]
    }
    consumer_ids = {
        gpu
        for r in rows
        if r["event"] == "consumer_launcher"
        for gpu in r["ray_gpu_ids"]
    }
    if len(producer_ids) != 4 or len(consumer_ids) != 4 or producer_ids & consumer_ids:
        raise RuntimeError("Observed producer and consumer GPU assignments are invalid")
    starts = {
        r["position"]: r["monotonic"] for r in rows if r["event"] == "inference_start"
    }
    intervals = [
        (starts[r["position"]], r["monotonic"])
        for r in rows
        if r["event"] == "inference_end"
    ]
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
        "producer_gpu_ids": sorted(producer_ids),
        "consumer_gpu_ids": sorted(consumer_ids),
        "inference_compute_overlap_seconds": overlap,
        "transfer_compute_overlap_seconds": transfer_overlap,
        "backpressure_events": sum(r["event"] == "backpressure" for r in rows),
        "verified_rank_receives": sum(r["event"] == "received" for r in rows),
    }


def launch(config, config_path):
    import mooncake
    import ray
    from ray.util.placement_group import placement_group, remove_placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    from .actors import Consumer, Producer
    from .buffer import FeatureBuffer

    output = Path(config["output_dir"])
    write_json(
        output / "environment.json",
        {
            "hostname": socket.gethostname(),
            "python": sys.executable,
            "versions": {
                p: importlib.metadata.version(p)
                for p in ("torch", "vllm", "ray", "mooncake-transfer-engine")
            },
            "gpu_inventory": require_idle_gpus(),
            "source_sha256": {
                str(path.relative_to(ROOT)): hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
                for path in [
                    *sorted((ROOT / "deepspec/pipeline").glob("*.py")),
                    ROOT / "torchtitan/torchtitan/trainer.py",
                    ROOT / "torchtitan/torchtitan/models/dspark_draft/data.py",
                ]
            },
        },
    )
    master_log = (output / "mooncake-master.log").open("w")
    master = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "deepspec.orchestration.process",
            str(Path(mooncake.__file__).parent / "mooncake_master"),
            f"--rpc_port={config['store']['master'].rsplit(':', 1)[1]}",
            f"--metrics_port={free_port()}",
            "--enable_offload=false",
            "--enable_disk_eviction=false",
        ],
        env=dict(
            os.environ,
            **environment(config["model_path"]),
            DEEPSPEC_ORCHESTRATOR_PID=str(os.getpid()),
        ),
        stdout=master_log,
        stderr=subprocess.STDOUT,
    )
    actors, groups = [], []
    buffer = producer = None
    try:
        deadline = time.monotonic() + 30
        while True:
            try:
                with socket.create_connection(
                    (
                        config["store"]["host"],
                        int(config["store"]["master"].rsplit(":", 1)[1]),
                    ),
                    timeout=1,
                ):
                    break
            except OSError:
                if master.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("Mooncake master did not start")
                time.sleep(0.1)
        os.environ.update(environment(config["model_path"]))
        context = ray.init(
            address="local",
            num_gpus=8,
            num_cpus=24,
            namespace=config["namespace"],
            include_dashboard=False,
            object_store_memory=128 * 1024**2,
            log_to_driver=False,
            _temp_dir=tempfile.mkdtemp(prefix="dspark-ray-", dir="/tmp"),
        )
        config["ray_address"] = context.address_info["gcs_address"]
        config["ray_logs"] = context.address_info["session_dir"] + "/logs"
        write_json(config_path, config)
        common_env = environment(config["model_path"])
        buffer = (
            ray.remote(FeatureBuffer)
            .options(
                name=config["buffer_name"],
                num_cpus=1,
                num_gpus=0,
                max_restarts=0,
                runtime_env={"env_vars": common_env},
            )
            .remote(config)
        )
        actors.append(buffer)
        ray.get(buffer.summary.remote(), timeout=60)
        # Four one-GPU worker bundles plus a CPU-only frontend. vLLM creates
        # the actual GPU actors itself in these bundles (no double reservation).
        producer_group = placement_group(
            [{"GPU": 1}] * 4 + [{"CPU": 1}], strategy="STRICT_PACK"
        )
        groups.append(producer_group)
        ray.get(producer_group.ready(), timeout=60)
        consumer_group = placement_group([{"GPU": 4, "CPU": 8}], strategy="STRICT_PACK")
        groups.append(consumer_group)
        ray.get(consumer_group.ready(), timeout=60)
        producer = (
            ray.remote(Producer)
            .options(
                num_cpus=1,
                num_gpus=0,
                max_restarts=0,
                runtime_env={
                    "env_vars": {
                        **common_env,
                        "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                    }
                },
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=producer_group,
                    placement_group_bundle_index=4,
                    placement_group_capture_child_tasks=True,
                ),
            )
            .remote(str(config_path))
        )
        actors.append(producer)
        consumer = (
            ray.remote(Consumer)
            .options(
                num_cpus=8,
                num_gpus=4,
                max_restarts=0,
                runtime_env={"env_vars": common_env},
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=consumer_group, placement_group_bundle_index=0
                ),
            )
            .remote(str(config_path))
        )
        actors.append(consumer)
        pending = {producer.run.remote(): "producer", consumer.run.remote(): "consumer"}
        results = {}
        deadline = time.monotonic() + config["timeout_seconds"]
        while pending:
            finished, _ = ray.wait(list(pending), timeout=5)
            check_gpu_ownership(output)
            for ref in finished:
                role = pending.pop(ref)
                results[role] = ray.get(ref)
                print(f"{role} completed", flush=True)
            if time.monotonic() > deadline:
                raise TimeoutError("Pipeline run timed out")
        results["buffer"] = ray.get(buffer.summary.remote())
        if results["buffer"]["remaining"] or results["buffer"]["released"] != len(
            config["samples"]
        ):
            raise RuntimeError("The pipeline did not drain every feature object")
        results["events"] = summarize_events(Path(config["events_path"]), config)
        from torchtitan.models.dspark_draft.checkpoint import read_commit

        commit = read_commit(results["consumer"]["commit"]["checkpoint"])
        if (
            commit != results["consumer"]["commit"]
            or commit["completed_updates"] != config["steps"]
            or commit["next_global_microbatch"] != len(config["samples"])
            or commit["run_id"] != config["run_id"]
        ):
            raise RuntimeError(
                "Persisted checkpoint does not match the completed stream"
            )
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
                logger.exception("Could not report failure to buffer")
        raise
    finally:
        if producer is not None:
            try:
                ray.get(producer.close.remote(), timeout=15)
            except Exception:
                logger.exception(
                    "Producer shutdown did not complete before actor cleanup"
                )
        # Killing the consumer launcher triggers its existing subreaper to stop
        # its torchrun descendants. Never stop unrelated jobs or Ray clusters.
        for actor in reversed(actors):
            if actor == buffer:
                try:
                    ray.get(buffer.close.remote(), timeout=10)
                except Exception:
                    logger.exception("Buffer close failed during cleanup")
            ray.kill(actor, no_restart=True)
        for group in groups:
            remove_placement_group(group)
        ray.shutdown()
        master.terminate()
        try:
            master.wait(timeout=10)
        except subprocess.TimeoutExpired:
            master.kill()
            master.wait()
        master_log.close()


def main():
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
    parser.add_argument("--protocol", choices=("tcp", "rdma"), default="tcp")
    parser.add_argument("--rdma-devices", default="")
    parser.add_argument("--receive-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
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
        "samples_per_update": 4,
        "consumer_world_size": 4,
        "window": args.window,
        "pool_bytes": args.pool_gib * GIB,
        "timeout_seconds": args.timeout_seconds,
        "receive_device": args.receive_device,
        "verify_transfers": True,
        "events_path": str(output / "events.jsonl"),
        "store": {
            "host": host,
            "master": f"{host}:{free_port()}",
            "protocol": args.protocol,
            "rdma_devices": args.rdma_devices,
        },
    }
    path = output / "pipeline.json"
    write_json(path, config)
    prepare(config, path)
    print(f"Prepared {len(config['samples'])} samples in {output}", flush=True)
    if not args.prepare_only:
        launch(config, path)


if __name__ == "__main__":
    main()
