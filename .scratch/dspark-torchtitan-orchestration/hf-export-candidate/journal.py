"""Reconcile orchestration progress with durable native commits."""

import json
import os
from pathlib import Path
import subprocess

from .io import atomic_json, digest
from .process import run_owned


def recover_commit(root, *, plan_path, workers):
    plan = json.loads(Path(plan_path).read_text())
    candidates = []
    for path in (Path(root) / "checkpoints").glob("step-*/commit.json"):
        if not path.is_file() or not (path.parent / ".metadata").is_file():
            continue
        commit = json.loads(path.read_text())
        if commit["metadata_sha256"] != digest(path.parent / ".metadata"):
            raise ValueError(f"Committed checkpoint metadata changed: {path.parent}")
        step = commit["completed_updates"]
        if (
            commit["format_version"] != 1
            or commit["run_id"] != plan["run_id"]
            or commit["input_plan_identity"] != digest(plan_path)
            or commit["world_size"] != workers
            or commit["next_global_microbatch"]
            != step * plan["gradient_accumulation_steps"]
            or not 0 < step <= plan["training_steps"]
            or path.parent.name != f"step-{step}"
            or Path(commit["checkpoint"]).resolve() != path.parent.resolve()
        ):
            raise ValueError(
                f"Checkpoint does not belong to this run and plan: {path.parent}"
            )
        candidates.append(commit)
    latest = max(candidates, key=lambda value: value["completed_updates"], default=None)
    progress = Path(root) / "progress.json"
    if progress.exists():
        recorded = json.loads(progress.read_text())
        if recorded["completed_updates"] > (
            latest["completed_updates"] if latest else 0
        ):
            raise ValueError("Recorded progress has no matching complete checkpoint")
    atomic_json(
        progress,
        {
            "completed_updates": latest["completed_updates"] if latest else 0,
            "checkpoint": latest["checkpoint"] if latest else None,
            "plan_identity": digest(plan_path),
        },
    )
    return latest


def validate_producer(result, requirements):
    path = Path(result["producer_manifest"])
    if digest(path) != result["producer_sha256"]:
        raise ValueError("Ready producer manifest changed")
    manifest = json.loads(path.read_text())
    for key, value in requirements.items():
        if manifest["teacher"][key] != value:
            raise ValueError(f"Ready producer {key} differs from the input plan")
    for sample in manifest["samples"]:
        for shard in sample["shards"]:
            if digest(path.parent / shard["path"]) != shard["sha256"]:
                raise ValueError("Ready producer shard is incomplete or changed")
    return manifest


def finish_pending_exports(request, commit):
    if commit is None:
        return
    config = commit["resolved_recipe"]["checkpoint"]
    points = {
        step
        for step in config.get("export_hf_steps", [])
        if step <= commit["completed_updates"]
    }
    final = commit["resolved_recipe"]["training"]["steps"]
    if config.get("export_hf_final") and commit["completed_updates"] == final:
        points.add(final)
    folder = Path(commit["checkpoint"]).parent
    for step in sorted(points):
        output = folder / "hf" / f"step-{step}"
        marker = output / "export.json"
        complete = False
        if marker.is_file():
            try:
                export = json.loads(marker.read_text())
                saved = export["checkpoint"]
                index = output / "model.safetensors.index.json"
                names = (
                    set(json.loads(index.read_text())["weight_map"].values())
                    if index.is_file()
                    else {"model.safetensors"}
                )
                complete = (
                    export.get("export_dtype") == config["export_dtype"]
                    and saved["run_id"] == commit["run_id"]
                    and saved["completed_updates"] == step
                    and (output / "config.json").is_file()
                    and all(
                        (output / name).is_file() and (output / name).stat().st_size > 0
                        for name in names
                    )
                )
            except (ValueError, KeyError):
                pass
        if complete:
            continue
        source = folder / f"step-{step}"
        if not (source / "commit.json").is_file():
            raise ValueError(
                f"Requested HF export {step} has neither weights nor a retained DCP"
            )
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        environment["PYTHONPATH"] = os.pathsep.join(
            [request["draft_source"], environment.get("PYTHONPATH", "")]
        )
        with (folder / f"export-{step}.log").open("a") as log:
            run_owned(
                [
                    request["draft_python"],
                    "-m",
                    "torchtitan.models.dspark_draft.export",
                    str(source),
                    str(output),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
