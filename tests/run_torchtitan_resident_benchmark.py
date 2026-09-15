"""Measure the retained resident loop on the exact completed 128K workload."""

import argparse
import json
import os
from pathlib import Path
import time

from deepspec.orchestration.io import atomic_json, digest, require_idle
from deepspec.orchestration.process import run_owned


def run(source, output):
    preparation_started = time.monotonic()
    original = json.loads((source / "run.json").read_text())
    summary = json.loads((source / "complete.json").read_text())
    plan_path = source / "inputs/input-plan.json"
    plan = json.loads(plan_path.read_text())
    initialization = Path(plan["resolved_recipe"]["capture_initialization"])
    if not (initialization / "initial-weights.pt").is_file():
        raise ValueError("The original workload must capture its initialization")
    if summary["completed_updates"] != 10 or summary["plan_identity"] != digest(
        plan_path
    ):
        raise ValueError("The original full workload must complete before comparison")
    phases = sorted(
        summary["phases"], key=lambda value: value["draft"]["consumed_range"][0]
    )
    request = {
        "plan_path": str(plan_path),
        "initialization": str(initialization),
        "output_dir": str(output),
        "partitions": [
            {
                "manifest": phase["draft"]["commit"]["resolved_recipe"]["dataloader"][
                    "manifest"
                ],
                "microbatch_start": phase["draft"]["consumed_range"][0],
            }
            for phase in phases
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    request_path = output / "request.json"
    atomic_json(request_path, request)
    environment = os.environ.copy()
    devices = original["draft_devices"]
    # Match the physical devices of native DP owners, retaining logical DP2.
    environment["CUDA_VISIBLE_DEVICES"] = ",".join((devices[0], devices[4]))
    command = [
        original["draft_python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "tests.benchmark_torchtitan_resident",
        str(request_path),
    ]
    preparation_seconds = time.monotonic() - preparation_started
    before = require_idle(devices)
    started = time.monotonic()
    run_owned(command, env=environment)
    result = json.loads((output / "result.json").read_text())
    for pid in result["worker_pids"]:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError(f"Resident benchmark worker {pid} is still alive")
    finished = time.monotonic()
    timing = result["timing"]
    timing["launch_seconds"] = timing["started_monotonic"] - started
    timing["exit_seconds"] = finished - timing["finished_monotonic"]
    release_started = time.monotonic()
    after = require_idle(devices)
    release_seconds = time.monotonic() - release_started
    result.update(
        {
            "resident_elapsed_seconds": finished - started,
            "resident_preparation_seconds": preparation_seconds,
            "resident_release_seconds": release_seconds,
            "resident_total_seconds": preparation_seconds
            + finished
            - started
            + release_seconds,
            "before_draft": before,
            "after_draft": after,
            "available_devices": devices,
            "used_devices": [devices[0], devices[4]],
            "unused_devices": [
                device for index, device in enumerate(devices) if index not in (0, 4)
            ],
            "plan_identity": digest(plan_path),
            "reference_kind": "retained resident trainer, logical DP2, physical FSDP2",
        }
    )
    atomic_json(output / "complete.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.source.resolve(), args.output.resolve())))
