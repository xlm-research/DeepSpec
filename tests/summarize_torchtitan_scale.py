"""Validate and summarize completed real 128K phase benchmark artifacts."""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import torch

from deepspec.orchestration.io import digest
from tests.compare_torchtitan_checkpoints import equal


def phases_at(root):
    summary = json.loads((root / "complete.json").read_text())
    phases = sorted(
        summary["phases"], key=lambda phase: phase["draft"]["consumed_range"][0]
    )
    cursor = 0
    for phase in phases:
        draft = phase["draft"]
        start, stop = draft["consumed_range"]
        if start != cursor or stop <= start or stop % 2:
            raise ValueError("The phases do not cover complete updates in order")
        commit = draft["commit"]
        recipe = commit["resolved_recipe"]
        plan_path = Path(recipe["dataloader"]["plan_path"])
        plan = json.loads(plan_path.read_text())
        if (
            digest(plan_path) != commit["input_plan_identity"]
            or len(plan["batches"]) != 40
            or any(batch["length"] != 131072 for batch in plan["batches"])
            or plan["global_batch_size"] != 4
            or plan["gradient_accumulation_steps"] != 2
            or plan["training_steps"] != 10
        ):
            raise ValueError(
                "The actual input plan differs from the fixed 128K workload"
            )
        model = recipe["model_spec"]["model"]["hf_config"]
        expected_model = {
            "num_hidden_layers": 5,
            "hidden_size": 5120,
            "intermediate_size": 17408,
            "vocab_size": 248320,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "num_anchors": 512,
            "block_size": 7,
            "markov_rank": 256,
        }
        if any(model[key] != value for key, value in expected_model.items()) or any(
            recipe["training"][key] != value
            for key, value in {
                "max_context_length": 131072,
                "num_tokens_per_microbatch_per_dp_rank": 131072,
                "num_tokens_per_train_step": 524288,
                "steps": 10,
            }.items()
        ):
            raise ValueError(
                "The completed phases do not use the fixed full 128K workload"
            )
        if (
            len(draft["worker_pids"]) != 8
            or len(set(draft["worker_pids"])) != 8
            or commit["world_size"] != 8
            or commit["completed_updates"] * 2 != stop
            or commit["metadata_sha256"]
            != digest(Path(commit["checkpoint"]) / ".metadata")
        ):
            raise ValueError("A phase lacks its complete eight-rank committed state")
        for boundary in ("before_draft", "after_draft"):
            observation = phase[boundary]
            if len(observation["devices"]) != 8 or observation["compute_processes"]:
                raise ValueError("A phase did not release the entire GPU pool")
        cursor = stop
    if cursor != 20:
        raise ValueError("The full workload must complete ten GAS-two updates")
    return phases


def phase_measurements(phase):
    draft = phase["draft"]
    timing = draft["timing"]
    total = draft["draft_elapsed_seconds"]
    if "draft_total_seconds" in draft and not math.isclose(
        draft["draft_total_seconds"],
        draft["draft_preparation_seconds"] + total + draft["draft_release_seconds"],
        abs_tol=1e-6,
    ):
        raise ValueError(
            "Full draft time must include feature preparation and GPU release"
        )
    if not math.isclose(
        total,
        sum(
            timing[key] for key in ("launch_seconds", "native_seconds", "exit_seconds")
        ),
        abs_tol=1e-6,
    ):
        raise ValueError("The parent timeline does not account for phase wall time")
    ranks = timing["ranks"]
    if {rank["rank"] for rank in ranks} != set(range(8)):
        raise ValueError("Timing must contain all eight actual ranks")
    for rank in ranks:
        end = 0
        for event in rank["events"]:
            if event["start_seconds"] < end - 1e-6 or event["seconds"] < 0:
                raise ValueError("Rank timing events overlap or have negative duration")
            end = event["start_seconds"] + event["seconds"]
    longest = max(ranks, key=lambda rank: rank["native_seconds"])
    breakdown = defaultdict(float)
    for event in longest["events"]:
        breakdown[event["name"]] += event["seconds"]
    breakdown["other_native"] = longest["native_seconds"] - sum(breakdown.values())
    training = []
    for event in longest["events"]:
        if event["name"] == "training":
            training.append({"step": event["step"], "seconds": event["seconds"]})
    return {
        "consumed_microbatch_range": draft["consumed_range"],
        "wall_seconds": total,
        "preparation_seconds": draft.get("draft_preparation_seconds"),
        "release_seconds": draft.get("draft_release_seconds"),
        "total_seconds": draft.get("draft_total_seconds"),
        "launch_seconds": timing["launch_seconds"],
        "native_seconds": timing["native_seconds"],
        "exit_seconds": timing["exit_seconds"],
        "longest_rank": longest["rank"],
        "longest_rank_seconds": longest["native_seconds"],
        "longest_rank_breakdown": dict(breakdown),
        "time_outside_longest_rank": timing["native_seconds"]
        - longest["native_seconds"],
        "training_updates_on_longest_rank": training,
        "compile_threads": sorted({rank["compile_threads"] for rank in ranks}),
        "peak_allocated_bytes": max(rank["peak_allocated_bytes"] for rank in ranks),
        "peak_reserved_bytes": max(rank["peak_reserved_bytes"] for rank in ranks),
    }


def supervision_at(root, phases):
    observations = [[] for _ in range(8)]
    for phase in phases:
        draft = phase["draft"]
        manifest = Path(draft["commit"]["resolved_recipe"]["dataloader"]["manifest"])
        accepted = json.loads((manifest.parent / "draft-result.json").read_text())
        if accepted["worker_pids"] != draft["worker_pids"]:
            raise ValueError("The observation directory belongs to a different phase")
        for rank in range(8):
            observations[rank].extend(
                torch.load(
                    manifest.parent / f"supervision-rank{rank}.pt", weights_only=True
                )
            )
    for rank in observations:
        if [value["next_microbatch"] for value in rank] != list(range(1, 21)):
            raise ValueError(
                "Supervision observations have missing or repeated samples"
            )
    return observations


def summarize(root, reference=None, resident=None):
    completed = json.loads((root / "complete.json").read_text())
    phases = phases_at(root)
    measurements = [phase_measurements(phase) for phase in phases]
    observations = supervision_at(root, phases)
    if any(phase["total_seconds"] is None for phase in measurements):
        raise ValueError(
            "The full benchmark must include orchestration feature validation"
        )
    result = {
        "root": str(root),
        "completed_updates": 10,
        "gas": 2,
        "actual_workers": 8,
        "phases": measurements,
        "draft_wall_seconds": sum(phase["wall_seconds"] for phase in measurements),
        "shared_preparation_seconds": completed.get("shared_preparation_seconds", 0),
        "draft_total_seconds": completed.get("shared_preparation_seconds", 0)
        + sum(phase["total_seconds"] for phase in measurements),
        "observed_microbatches_per_rank": [len(rank) for rank in observations],
        "resource_boundaries_verified": True,
        "timing_note": (
            "Each native breakdown follows one actual rank's nonoverlapping timeline. "
            "Time outside that rank completes the all-rank native wall span. "
            "Compiler durations are nested in training and are not added again."
        ),
    }
    if all("producer" in phase for phase in phases):
        result["excluded_target_seconds"] = sum(
            json.loads(Path(phase["producer"]["producer_manifest"]).read_text())[
                "elapsed_seconds"
            ]
            for phase in phases
        )
    if reference is not None:
        expected = supervision_at(reference, phases_at(reference))
        if not equal(observations, expected):
            raise ValueError("Supervision, sample order or per-forward RNG differs")
        result["supervision_reference"] = str(reference)
        result["supervision_bitwise_equal"] = True
    if resident is not None:
        resident_result = json.loads((resident / "complete.json").read_text())
        if (
            resident_result["completed_updates"] != 10
            or resident_result["actual_workers"] != 2
        ):
            raise ValueError(
                "The resident comparison did not complete the same workload"
            )
        expected = [
            torch.load(resident / f"observations-rank{rank}.pt", weights_only=True)[
                "supervision"
            ]
            for rank in range(2)
        ]
        if not equal([observations[0], observations[4]], expected):
            raise ValueError("The resident and native supervision or RNG differs")
        result["resident_reference"] = str(resident)
        result["resident_supervision_bitwise_equal"] = True
    if torch.cuda.is_initialized():
        raise RuntimeError("Artifact validation unexpectedly initialized CUDA")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resident", type=Path)
    args = parser.parse_args()
    report = summarize(args.root.resolve(), args.reference, args.resident)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))
