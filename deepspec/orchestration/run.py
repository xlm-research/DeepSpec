"""Alternate target production and native draft phases over one GPU pool."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from .draft import run_draft_phase
from .io import atomic_json, digest, require_idle
from .process import run_owned
from .journal import finish_pending_exports, recover_commit, validate_producer


def run(request):
    root = Path(request["output_dir"]).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "owner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = root / "run.json"
        if saved.exists():
            if json.loads(saved.read_text()) != request:
                raise ValueError(
                    "Resume request differs from the recorded orchestration run"
                )
        else:
            if (root / "inputs/input-plan.json").exists():
                raise ValueError("Existing plan has no owning orchestration request")
            atomic_json(saved, request)
        return _run(request, root)


def _run(request, root):
    require_idle(request["producer"]["devices"])
    preparation = {
        "run_id": request["run_id"],
        "recipe_args": request["recipe_args"],
        "workers": request["workers"],
        "output_dir": str(root / "inputs"),
    }
    prep_request = root / "prepare.json"
    atomic_json(prep_request, preparation)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [request["draft_source"], environment.get("PYTHONPATH", "")]
    )
    environment["CUDA_VISIBLE_DEVICES"] = ""
    plan_path = root / "inputs/input-plan.json"
    if not plan_path.is_file():
        if (root / "inputs").exists():
            (root / "inputs").rename(root / f"abandoned-inputs-{time.time_ns()}")
        with (root / "prepare.log").open("a") as log:
            run_owned(
                [
                    request["draft_python"],
                    "-m",
                    "torchtitan.models.dspark_draft.preparation",
                    str(prep_request),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
    plan = json.loads(plan_path.read_text())
    gas = plan["gradient_accumulation_steps"]
    global_batch = plan["global_batch_size"]
    steps_per_partition = int(request["updates_per_partition"])
    if steps_per_partition < 1:
        raise ValueError("Partition size must contain complete updates")
    pool = request["producer"]["devices"]
    if (
        not set(request["draft_devices"]).issubset(set(pool))
        or len(request["draft_devices"]) != request["workers"]
    ):
        raise ValueError("Draft workers must use the assigned GPU pool")
    recovered = recover_commit(root, plan_path=plan_path, workers=request["workers"])
    finish_pending_exports(request, recovered)
    resume = recovered["checkpoint"] if recovered else None
    start = recovered["completed_updates"] if recovered else 0
    if not request.get("retain_features", False):
        reclaim_features(root, start)
    results = [
        json.loads(path.read_text())
        for path in sorted(root.glob("phase-*/complete.json"))
    ]
    while start < plan["training_steps"]:
        partition_start = start // steps_per_partition * steps_per_partition
        stop = min(partition_start + steps_per_partition, plan["training_steps"])
        phase_root = root / f"phase-{partition_start}-{stop}"
        phase_root.mkdir(exist_ok=True)
        producer_request = {
            "plan_path": str(plan_path),
            "producer": request["producer"],
            "start_position": partition_start * global_batch,
            "end_position": stop * global_batch,
            "output_dir": str(phase_root / "target"),
            "result_path": str(phase_root / "target-result.json"),
        }
        target_request_path = phase_root / "target-request.json"
        atomic_json(target_request_path, producer_request)
        require_idle(pool)
        if not (phase_root / "target-result.json").is_file():
            if (phase_root / "target").exists():
                (phase_root / "target").rename(
                    phase_root / f"abandoned-target-{time.time_ns()}"
                )
            with (phase_root / "target.log").open("a") as log:
                run_owned(
                    [
                        request["producer"]["config"]["python_executable"],
                        "-m",
                        "deepspec.orchestration.target",
                        str(target_request_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        require_idle(pool)
        preparation_started = time.monotonic()
        producer_result = json.loads((phase_root / "target-result.json").read_text())
        validate_producer(producer_result, plan["producer_requirements"])
        attempt = phase_root / f"attempt-{time.time_ns()}"
        attempt.mkdir()
        manifest = attempt / "consumption.json"
        atomic_json(
            manifest,
            {
                **producer_result,
                "batches": [
                    {"id": entry["id"]}
                    for entry in plan["batches"][
                        start * global_batch : stop * global_batch
                    ]
                ],
            },
        )
        draft_request = {
            key: request[key]
            for key in ("draft_python", "draft_source", "workers", "recipe_args")
        }
        draft_request["devices"] = request["draft_devices"]
        draft_request["result_path"] = str(attempt / "draft-result.json")
        draft_request["phase"] = {
            "run_id": request["run_id"],
            "stop_update": stop,
            "microbatch_start": start * gas,
            "feature_manifest": str(manifest),
            "plan_path": str(plan_path),
            "checkpoint_folder": str(root / "checkpoints"),
            "resume_checkpoint": resume,
        }
        atomic_json(attempt / "draft-request.json", draft_request)
        preparation_seconds = time.monotonic() - preparation_started
        before = require_idle(pool)
        result = run_draft_phase(draft_request)
        release_started = time.monotonic()
        after = require_idle(pool)
        result["draft_release_seconds"] = time.monotonic() - release_started
        result["draft_preparation_seconds"] = preparation_seconds
        result["draft_total_seconds"] = (
            preparation_seconds
            + result["draft_elapsed_seconds"]
            + result["draft_release_seconds"]
        )
        resume = result["commit"]["checkpoint"]
        phase_result = {
            "draft": result,
            "before_draft": before,
            "after_draft": after,
            "producer": producer_result,
        }
        atomic_json(phase_root / "complete.json", phase_result)
        atomic_json(
            root / "progress.json",
            {
                "completed_updates": stop,
                "checkpoint": resume,
                "plan_identity": digest(plan_path),
            },
        )
        if not request.get("retain_features", False):
            reclaim_features(root, stop)
        results.append(phase_result)
        start = stop
    summary = {
        "run_id": request["run_id"],
        "plan_identity": digest(plan_path),
        "phases": results,
        "completed_updates": start,
        "recovered_commit": recovered,
    }
    atomic_json(root / "complete.json", summary)
    return summary


def reclaim_features(root, committed_update):
    for phase in root.glob("phase-*-*"):
        _, start, stop = phase.name.split("-")
        if not (start.isdigit() and stop.isdigit()) or int(stop) > committed_update:
            continue
        for features in [
            phase / "target/features",
            *phase.glob("abandoned-target-*/features"),
        ]:
            if features.is_dir():
                shutil.rmtree(features)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.request.read_text()))))
