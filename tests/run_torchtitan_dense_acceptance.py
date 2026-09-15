"""Full-size Qwen draft checks on short windows of archived teacher features."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from deepspec.orchestration.io import require_idle
from tests.compare_torchtitan_checkpoints import compare
from torchtitan.models.dspark_draft.planning import make_input_plan


def prepare(source, output):
    plan_bytes = (source / "inputs/input-plan.json").read_bytes()
    original = json.loads(plan_bytes)
    initialization = Path(original["resolved_recipe"]["capture_initialization"])
    weights = initialization / "initial-weights.pt"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if output.exists():
        windows = json.loads((output / "windows.json").read_text())
        if windows["source_plan_sha256"] != hashlib.sha256(plan_bytes).hexdigest():
            raise ValueError("Prepared windows belong to another source plan")
        plan = json.loads((output / "input-plan.json").read_text())
        names = [entry["id"] for entry in plan["batches"]]
        if len(names) != 32 or any(not (output / name).is_file() for name in names):
            raise ValueError("Prepared windows are incomplete")
        return weights, names
    output.mkdir(parents=True, exist_ok=False)
    batches = []
    windows = []
    for position, entry in enumerate(original["batches"][:32]):
        phase = "phase-0-5" if position < 20 else "phase-5-10"
        path = source / phase / "target/features" / entry["id"]
        full = torch.load(path, weights_only=True, mmap=True, map_location="cpu")
        enabled = full["loss_mask"][0].nonzero().flatten()
        if not enabled.numel():
            raise ValueError(f"No supervised tokens in {path}")
        start = max(0, int(enabled[0]) - 32)
        stop = start + 128
        if stop > full["input_ids"].shape[1]:
            raise ValueError(f"No complete short window in {path}")
        batch = {
            key: full[key][:, start:stop].clone()
            for key in (
                "input_ids",
                "loss_mask",
                "target_hidden_states",
                "target_last_hidden_states",
            )
        }
        batch["context_chunk_len"] = torch.tensor([128])
        batch["seq_len"] = torch.tensor([128])
        name = f"batch-{position // 8}-rank{position % 8}.pt"
        torch.save(batch, output / name)
        batches.append((name, batch))
        windows.append({"id": name, "source": str(path), "start": start, "stop": stop})
    if len(batches) != 32:
        raise ValueError("Two DP8/GAS2 updates require 32 samples")
    plan = make_input_plan(run_id="qwen38-dense-short", ordered_batches=batches)
    (output / "input-plan.json").write_text(json.dumps(plan, sort_keys=True))
    (output / "windows.json").write_text(
        json.dumps(
            {
                "source_plan_sha256": hashlib.sha256(plan_bytes).hexdigest(),
                "windows": windows,
                "position_convention": "Positions restart at zero in each 128-token window",
            },
            indent=2,
        )
    )
    return weights, [name for name, _ in batches]


def run(source, output):
    devices = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
    if len(devices) != 8:
        raise ValueError("This acceptance requires eight physical GPUs")
    require_idle(devices)
    weights, names = prepare(source, output)
    reports = {}
    for topology in ("shard8", "replicate8"):
        phases = []
        for name, start, stop in (
            ("continuous", 0, 2),
            ("first", 0, 1),
            ("resumed", 1, 2),
        ):
            root = output / topology / name
            root.mkdir(parents=True)
            manifest = root / "features.json"
            manifest.write_text(
                json.dumps(
                    {
                        "batches": [
                            {"id": entry, "path": str(output / entry)}
                            for entry in names[start * 16 : stop * 16]
                        ]
                    }
                )
            )
            request = {
                "draft_python": sys.executable,
                "draft_source": str(Path("torchtitan").resolve()),
                "workers": 8,
                "devices": devices,
                "result_path": str(root / "result.json"),
                "recipe_args": [
                    "--module",
                    "torchtitan.models.dspark_draft.config_registry",
                    "--config",
                    f"qwen38_27b_{topology}",
                    "--initial-weights",
                    str(weights),
                    "--training.steps",
                    "2",
                    "--training.max-context-length",
                    "128",
                    "--training.num-tokens-per-microbatch-per-dp-rank",
                    "128",
                    "--training.num-tokens-per-train-step",
                    "2048",
                    "--dataloader.no-require-producer-manifest",
                    "--metrics.save-for-all-ranks",
                    "--metrics.log-freq",
                    "1",
                    "--measure-phase",
                    "--dump-folder",
                    str(root / "native-logs"),
                ],
                "phase": {
                    "run_id": "qwen38-dense-short",
                    "stop_update": stop,
                    "feature_manifest": str(manifest),
                    "microbatch_start": start * 2,
                    "plan_path": str(output / "input-plan.json"),
                    "checkpoint_folder": str(root / "checkpoints"),
                    "resume_checkpoint": str(
                        output / topology / "first/checkpoints/step-1"
                    )
                    if start
                    else None,
                },
            }
            request_path = root / "request.json"
            request_path.write_text(json.dumps(request, indent=2))
            require_idle(devices)
            started = time.monotonic()
            print(f"Starting {topology}/{name}", flush=True)
            with (root / "native.log").open("w") as log:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "deepspec.orchestration.draft",
                        str(request_path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = json.loads((root / "native.log").read_text().splitlines()[-1])
            if result["completed_updates"] != stop or len(result["worker_pids"]) != 8:
                raise ValueError(
                    "The full-size phase did not complete on eight workers"
                )
            released = require_idle(devices)
            phase = {
                "name": name,
                "elapsed_seconds": time.monotonic() - started,
                "release": released,
                "result": result,
            }
            (root / "complete.json").write_text(json.dumps(phase, indent=2))
            phases.append(phase)
        comparison = compare(
            output / topology / "continuous/checkpoints/step-2",
            output / topology / "resumed/checkpoints/step-2",
        )
        (output / topology / "comparison.json").write_text(
            json.dumps(comparison, indent=2)
        )
        if not comparison["bitwise_equal"]:
            raise ValueError(f"{topology} restart differs: {comparison['differences']}")
        reports[topology] = {"phases": phases, "comparison": comparison}
        print(f"Passed {topology}: full checkpoint restart equality", flush=True)
    (output / "complete.json").write_text(json.dumps(reports, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args.source.resolve(), args.output.resolve())
        if torch.cuda.is_initialized():
            raise RuntimeError("Feature preparation unexpectedly initialized CUDA")
        print("Prepared 32 fixed-feature windows on CPU")
    else:
        run(args.source.resolve(), args.output.resolve())
