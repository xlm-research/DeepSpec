"""Schedule the existing, unchanged Qwen vLLM producer on prepared token inputs."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import time

from deepspec.trainer.qwen3_8_vllm import (
    QwenVllmConfig,
    run_worker_process,
    teacher_identity,
)

from .io import atomic_json, digest, require_idle


def produce_partition(request):
    started = time.monotonic()
    root = Path(request["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=False)
    plan = json.loads(Path(request["plan_path"]).read_text())
    producer = request["producer"]
    config = QwenVllmConfig(**producer["config"])
    devices = producer["devices"]
    replicas = producer["replicas"]
    if len(devices) != replicas * config.tensor_parallel_size:
        raise ValueError("Producer devices do not match its established DP/TP layout")
    before = require_idle(devices)
    teacher = teacher_identity(
        producer["model_path"], plan["producer_requirements"]["target_layer_ids"]
    )
    for key, value in plan["producer_requirements"].items():
        if teacher[key] != value:
            raise ValueError(
                f"Prepared input requirement {key} differs from the producer"
            )
    selected = plan["batches"][request["start_position"] : request["end_position"]]
    if (
        len(selected) != request["end_position"] - request["start_position"]
        or not selected
    ):
        raise ValueError("Requested producer range is outside the input plan")
    jobs = []
    samples = []
    for replica in range(replicas):
        entries = selected[replica::replicas]
        requests = []
        for sample in entries:
            output_path = root / "features" / sample["id"]
            requests.append(
                {"input_path": sample["input_path"], "output_paths": [str(output_path)]}
            )
            samples.append(
                {
                    "sample_id": sample["sample_id"],
                    "position": sample["position"],
                    "input_identity": sample["input_identity"],
                    "length": sample["length"],
                    "producer_replica": replica,
                    "shards": [{"cp_rank": 0, "path": str(output_path)}],
                }
            )
        if not requests:
            continue
        path = root / f"producer-{replica}.json"
        atomic_json(
            path,
            {
                "config": asdict(config),
                "teacher": teacher,
                "model_path": producer["model_path"],
                "max_length": producer["max_length"],
                "requests": requests,
            },
        )
        group = devices[
            replica * config.tensor_parallel_size : (replica + 1)
            * config.tensor_parallel_size
        ]
        jobs.append((path, group))
    with ThreadPoolExecutor(max_workers=replicas) as executor:
        futures = [
            executor.submit(run_worker_process, path, config, group)
            for path, group in jobs
        ]
        for future in futures:
            future.result()
    after = require_idle(devices)
    for path, _ in jobs:
        completion = json.loads(Path(str(path) + ".complete").read_text())
        if completion["teacher"] != teacher:
            raise ValueError("Producer completion differs from the planned teacher")
    for sample in samples:
        for shard in sample["shards"]:
            shard["sha256"] = digest(shard["path"])
    manifest = root / "producer.json"
    atomic_json(
        manifest,
        {
            "version": 1,
            "teacher": teacher,
            "config": asdict(config),
            "layout": {
                "replicas": replicas,
                "tp": config.tensor_parallel_size,
                "cp": 1,
            },
            "samples": sorted(samples, key=lambda sample: sample["position"]),
            "resources_before": before,
            "resources_after": after,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    result = {"producer_manifest": str(manifest), "producer_sha256": digest(manifest)}
    atomic_json(request["result_path"], result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(produce_partition(json.loads(args.request.read_text()))))
