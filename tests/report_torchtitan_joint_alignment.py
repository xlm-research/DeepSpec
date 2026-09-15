"""Report TP2/PP2 numerical differences, with or without CP, from CPU observations.

This does not change acceptance tolerances or launch any training workers.
"""

import argparse
import json
from pathlib import Path

import torch

from tests.compare_torchtitan_checkpoints import equal


def statistics(pairs):
    count = mismatches = 0
    maximum = square_error = square_reference = 0.0
    for actual, expected in pairs:
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise ValueError("Observation shape or dtype differs from reference")
        count += expected.numel()
        if not expected.is_floating_point():
            mismatches += int((actual != expected).sum())
            continue
        actual, expected = actual.double(), expected.double()
        delta = (actual - expected).abs()
        if not torch.isfinite(delta).all():
            raise ValueError("Non-finite numerical comparison")
        mismatches += int((delta > 1e-6 + 1e-4 * expected.abs()).sum())
        maximum = max(maximum, float(delta.max()))
        square_error += float(delta.square().sum())
        square_reference += float(expected.square().sum())
    return {
        "elements": count,
        "mismatched_elements": mismatches,
        "max_absolute_error": maximum,
        "relative_l2_error": (square_error / max(square_reference, 1e-300)) ** 0.5,
        "within_tolerance": mismatches == 0,
    }


@torch.no_grad()
def replay_adam(parts, initial_weights):
    """Independently replay this fixture's Adam formula on the observed gradients.

    This diagnoses update amplification; it does not replace reference checks.
    """
    maxima = {key: 0.0 for key in ("master_param", "exp_avg", "exp_avg_sq")}
    for part in parts:
        states = {
            name: [
                initial_weights[name].float().clone(),
                torch.zeros_like(gradient, dtype=torch.float32),
                torch.zeros_like(gradient, dtype=torch.float32),
            ]
            for name, gradient in part["updates"][0]["gradients_after_clip"].items()
        }
        for index, update in enumerate(part["updates"]):
            lr = (
                0.001 if index == 0
                else part["updates"][index - 1]["scheduler"]["_last_lr"][0]
            )
            step = index + 1
            for name, gradient in update["gradients_after_clip"].items():
                master, first, second = states[name]
                gradient = gradient.float()
                first = 0.9 * first + 0.1 * gradient
                second = 0.999 * second + 0.001 * gradient.square()
                master -= lr * (first / (1 - 0.9**step)) / (
                    (second / (1 - 0.999**step)).sqrt() + 1e-8
                )
                states[name] = [master, first, second]
                for key, value in zip(maxima, states[name], strict=True):
                    maxima[key] = max(
                        maxima[key], float((value - update["adam"][name][key]).abs().max())
                    )
    return {"maximum_absolute_errors": maxima, "replaces_reference_verdict": False}


def report(root, dtype, topology="tp_cp_pp"):
    cp = 2 if topology == "tp_cp_pp" else 1
    stage_size = 2 * cp
    fixture = torch.load(
        root / f"reference-dp1/torch.{dtype}_rank0.pt", weights_only=True
    )
    reference = fixture["result"]
    base = root / topology / dtype
    actual = [
        torch.load(base / "continuous" / f"rank-{rank}.pt", weights_only=True)
        for rank in range(2 * stage_size)
    ]
    rows = {}
    for index, expected in enumerate(reference["updates"]):
        for field in ("parameters", "gradients_after_clip", "adam"):
            owned = {}
            for rank in (0, stage_size):
                owned.update(actual[rank]["updates"][index][field])
            if owned.keys() != expected[field].keys():
                raise ValueError(f"Pipeline ownership differs for {field}")
            if field == "adam":
                for component in ("master_param", "exp_avg", "exp_avg_sq", "step"):
                    rows[f"update{index}/{component}"] = statistics(
                        (value[component], expected[field][name][component])
                        for name, value in owned.items()
                    )
            else:
                rows[f"update{index}/{field}"] = statistics(
                    (value, expected[field][name]) for name, value in owned.items()
                )
        rows[f"update{index}/norm"] = statistics(
            [(torch.tensor(actual[0]["grad_norm"][index]),
              torch.tensor(reference["metrics"]["grad_norm"][index]))]
        )
        if not all(
            equal(state["updates"][index]["scheduler"], expected["scheduler"])
            for state in actual
        ):
            raise ValueError("Scheduler states differ")
        for stage in (0, stage_size):
            if not all(
                equal(actual[rank]["updates"][index], actual[stage]["updates"][index])
                for rank in range(stage, stage + stage_size)
            ):
                raise ValueError("Restored full stage tensors differ across TP/CP ranks")
    for index, expected in enumerate(reference["microbatches"]):
        shards = [
            actual[rank]["microbatches"][index]
            for rank in range(stage_size, 2 * stage_size, 2)
        ]
        rows[f"microbatch{index}/loss"] = statistics(
            [(sum(shard["loss"] for shard in shards), expected["loss"] / 2)]
        )
        for term_index, name in enumerate(("ce_loss", "l1_loss", "confidence_loss")):
            numerator = sum(shard["terms"][f"{name}_num"] for shard in shards)
            denominator = sum(shard["terms"][f"{name}_den"] for shard in shards)
            rows[f"microbatch{index}/{name}"] = statistics(
                [(numerator / (denominator + 1e-6), expected["terms"][term_index])]
            )
            rows[f"microbatch{index}/{name}_denominator"] = statistics(
                [(denominator, expected["denominator"])]
            )
        for name, output in expected["output"].items():
            rows[f"microbatch{index}/{name}"] = statistics(
                [(torch.cat([shard["output"][name] for shard in shards], dim=1), output)]
            )
    restored = base / "resumed/complete.json"
    return {
        "dtype": dtype,
        "topology": f"TP2 x CP{cp} x PP2",
        "reference": "retained real five-layer Qwen DSpark, DP1/TP1/CP1/PP1",
        "rtol": 1e-4,
        "atol": 1e-6,
        "strict_numerical_match": all(row["within_tolerance"] for row in rows.values()),
        "comparisons": rows,
        "independent_adam_replay": replay_adam(
            [actual[0], actual[stage_size]], fixture["initial_weights"]
        ),
        "resume": json.loads(restored.read_text()) if restored.is_file() else None,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--topology", choices=("tp_cp_pp", "tp_pp"), default="tp_cp_pp")
    args = parser.parse_args()
    reports = {
        dtype: report(args.output, dtype, args.topology)
        for dtype in ("float32", "bfloat16")
    }
    destination = args.output / args.topology / "alignment-report.json"
    destination.write_text(json.dumps(reports, indent=2))
    print(destination)
    for dtype, result in reports.items():
        print(dtype, "strict_numerical_match:", result["strict_numerical_match"])
