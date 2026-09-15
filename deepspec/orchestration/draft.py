"""Launch a native draft recipe and return only after every worker exits."""

import argparse
import json
import os
from pathlib import Path
import time

from .process import run_owned


def run_draft_phase(request):
    result_path = Path(request["result_path"]).resolve()
    if result_path.exists():
        raise FileExistsError(f"Phase result already exists: {result_path}")
    source = Path(request["draft_source"]).resolve()
    if not (source / "torchtitan").is_dir():
        raise ValueError("draft_source must identify the TorchTitan checkout")
    workers = int(request["workers"])
    if workers < 1:
        raise ValueError("workers must be positive")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source), environment.get("PYTHONPATH", "")]
    )
    environment["DEEPSPEC_PHASE_RESULT"] = str(result_path)
    if "devices" in request:
        if len(request["devices"]) != workers:
            raise ValueError("Draft devices do not match the requested worker count")
        environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, request["devices"]))
    result_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        request["draft_python"],
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={workers}",
        "-m",
        "torchtitan.models.dspark_draft.train",
        *request["recipe_args"],
    ]
    phase = request.get("phase")
    if phase is not None:
        fields = {
            "run_id": "run-id",
            "stop_update": "phase-stop-update",
            "feature_manifest": "dataloader.manifest",
            "microbatch_start": "dataloader.global-microbatch-start",
            "plan_path": "dataloader.plan-path",
            "checkpoint_folder": "checkpoint.folder",
        }
        for key, flag in fields.items():
            command.extend([f"--{flag}", str(phase[key])])
        if phase.get("resume_checkpoint"):
            command.extend(
                ["--checkpoint.initial-load-path", phase["resume_checkpoint"]]
            )
    started = time.monotonic()
    run_owned(command, env=environment)
    result = json.loads(result_path.read_text())
    if phase is not None and (
        result["completed_updates"] != phase["stop_update"]
        or result["consumed_range"][0] != phase["microbatch_start"]
        or result["commit"]["run_id"] != phase["run_id"]
    ):
        raise ValueError("Draft result does not match the requested phase")
    for pid in result.get("worker_pids", []):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        raise RuntimeError(f"Draft worker {pid} is still alive after phase exit")
    if "commit" in result:
        finished = time.monotonic()
        result["draft_elapsed_seconds"] = finished - started
        if "timing" in result:
            timing = result["timing"]
            if "started_monotonic" in timing:
                timing["launch_seconds"] = timing["started_monotonic"] - started
                timing["exit_seconds"] = finished - timing["finished_monotonic"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_draft_phase(json.loads(args.request.read_text()))))
