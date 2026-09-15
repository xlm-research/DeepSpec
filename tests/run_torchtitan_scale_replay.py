"""Replay a fixed real feature workload through complete native draft phases."""

import argparse
import json
import os
from pathlib import Path
import time

from deepspec.orchestration.draft import run_draft_phase
from deepspec.orchestration.io import atomic_json, digest, require_idle
from deepspec.orchestration.journal import validate_producer


def run(source, output, partition_updates):
    preparation_started = time.monotonic()
    request = json.loads((source / "run.json").read_text())
    plan_path = source / "inputs/input-plan.json"
    plan = json.loads(plan_path.read_text())
    initialization = Path(plan["resolved_recipe"]["capture_initialization"])
    if not (initialization / "initial-weights.pt").is_file():
        raise ValueError(
            "The real validation run must first capture its initialization"
        )
    if sum(partition_updates) != plan["training_steps"] or any(
        count < 1 for count in partition_updates
    ):
        raise ValueError("Partitions must cover every complete update exactly once")
    setup_seconds = time.monotonic() - preparation_started
    require_idle(request["draft_devices"])
    preparation_started = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    producer = None
    sources = []
    samples = []
    for path in sorted(source.glob("phase-*/target-result.json")):
        result = json.loads(path.read_text())
        validate_producer(result, plan["producer_requirements"])
        manifest_path = Path(result["producer_manifest"])
        manifest = json.loads(manifest_path.read_text())
        facts = {
            key: manifest[key] for key in ("version", "teacher", "config", "layout")
        }
        if producer is None:
            producer = facts
        elif producer != facts:
            raise ValueError("The source partitions have different producer facts")
        for sample in manifest["samples"]:
            for shard in sample["shards"]:
                shard["path"] = str((manifest_path.parent / shard["path"]).resolve())
            samples.append(sample)
        sources.append(result)
    samples.sort(key=lambda value: value["position"])
    if [sample["sample_id"] for sample in samples] != [
        batch["sample_id"] for batch in plan["batches"]
    ]:
        raise ValueError("The source features do not cover the exact whole-run plan")
    producer_path = output / "producer.json"
    atomic_json(producer_path, {**producer, "samples": samples, "sources": sources})
    os.environ["DEEPSPEC_SCALE_CAPTURE"] = ""
    os.environ["DEEPSPEC_SCALE_REFERENCE"] = str(initialization)
    os.environ["DEEPSPEC_SCALE_OUTPUT"] = str(output)
    os.environ["DEEPSPEC_SCALE_DATA"] = plan["resolved_recipe"]["preparation"][
        "source_paths"
    ][0]
    phases = []
    start = 0
    checkpoint = None
    shared_preparation_seconds = setup_seconds + time.monotonic() - preparation_started
    for count in partition_updates:
        preparation_started = time.monotonic()
        stop = start + count
        root = output / f"phase-{start}-{stop}"
        root.mkdir()
        manifest = root / "consumption.json"
        batch_size = plan["global_batch_size"]
        atomic_json(
            manifest,
            {
                "producer_manifest": str(producer_path),
                "producer_sha256": digest(producer_path),
                "batches": [
                    {"id": batch["id"]}
                    for batch in plan["batches"][start * batch_size : stop * batch_size]
                ],
            },
        )
        phase = {
            **{
                key: request[key]
                for key in ("draft_python", "draft_source", "workers", "recipe_args")
            },
            # Preserve the native metrics peaks before each update resets them.
            # Metrics settings are outside the training-state identity.
            "recipe_args": [*request["recipe_args"], "--metrics.save-for-all-ranks"],
            "devices": request["draft_devices"],
            "result_path": str(root / "draft-result.json"),
            "phase": {
                "run_id": plan["run_id"],
                "stop_update": stop,
                "feature_manifest": str(manifest),
                "microbatch_start": start * plan["gradient_accumulation_steps"],
                "plan_path": str(plan_path),
                "checkpoint_folder": str(output / "checkpoints"),
                "resume_checkpoint": checkpoint,
            },
        }
        atomic_json(root / "request.json", phase)
        preparation_seconds = time.monotonic() - preparation_started
        before = require_idle(request["draft_devices"])
        result = run_draft_phase(phase)
        release_started = time.monotonic()
        after = require_idle(request["draft_devices"])
        result["draft_release_seconds"] = time.monotonic() - release_started
        result["draft_preparation_seconds"] = preparation_seconds
        result["draft_total_seconds"] = (
            preparation_seconds
            + result["draft_elapsed_seconds"]
            + result["draft_release_seconds"]
        )
        phases.append({"draft": result, "before_draft": before, "after_draft": after})
        atomic_json(root / "complete.json", phases[-1])
        checkpoint = result["commit"]["checkpoint"]
        start = stop
    summary = {
        "source": str(source),
        "plan_identity": digest(plan_path),
        "partition_updates": partition_updates,
        "phases": phases,
        "shared_preparation_seconds": shared_preparation_seconds,
        "draft_total_seconds": shared_preparation_seconds
        + sum(phase["draft"]["draft_total_seconds"] for phase in phases),
        "draft_elapsed_seconds": sum(
            phase["draft"]["draft_elapsed_seconds"] for phase in phases
        ),
    }
    atomic_json(output / "complete.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("partition_updates", type=int, nargs="+")
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.source.resolve(), args.output.resolve(), args.partition_updates)
        )
    )
