#!/usr/bin/env python3
"""Compare every stored GLM hidden-state value without loading both files at once."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open


def compare_layer(reference, candidate, *, column_start, width, label, block_rows):
    totals = dict(
        abs_error=0.0, squared_error=0.0, ref_squared=0.0, other_squared=0.0, dot=0.0
    )
    max_abs = 0.0
    exact = 0
    within = 0
    count = 0
    cosines = []
    blocks = []
    rows = reference.get_shape()[0]
    for start in range(0, rows, block_rows):
        end = min(rows, start + block_rows)
        a = reference[start:end, column_start : column_start + width].float()
        b = candidate[start:end, column_start : column_start + width].float()
        assert torch.isfinite(a).all() and torch.isfinite(b).all(), (label, start)
        delta = b - a
        absolute = delta.abs()
        a2 = a.square().sum(dim=-1, dtype=torch.float64)
        b2 = b.square().sum(dim=-1, dtype=torch.float64)
        dot = (a * b).sum(dim=-1, dtype=torch.float64)
        assert (a2 > 0).all() and (b2 > 0).all(), (label, "zero norm")
        cosine = (dot / (a2 * b2).sqrt()).clamp(-1, 1)
        cosines.append(cosine)
        sum_error = absolute.sum(dtype=torch.float64).item()
        sum_error2 = delta.square().sum(dtype=torch.float64).item()
        ref2 = a2.sum().item()
        other2 = b2.sum().item()
        totals["abs_error"] += sum_error
        totals["squared_error"] += sum_error2
        totals["ref_squared"] += ref2
        totals["other_squared"] += other2
        totals["dot"] += dot.sum().item()
        maximum = absolute.max().item()
        max_abs = max(max_abs, maximum)
        exact += (a == b).sum().item()
        within += (absolute <= (0.01 + 0.01 * a.abs())).sum().item()
        count += a.numel()
        blocks.append(
            {
                "layer": label,
                "token_start": start,
                "token_end": end,
                "mean_token_cosine": cosine.mean().item(),
                "mae": sum_error / a.numel(),
                "rmse": math.sqrt(sum_error2 / a.numel()),
                "relative_l2": math.sqrt(sum_error2 / ref2),
                "max_abs_error": maximum,
            }
        )
    cosine = torch.cat(cosines)
    result = {
        "layer": label,
        "shape": [rows, width],
        "elements_compared": count,
        "all_finite": True,
        "mae": totals["abs_error"] / count,
        "rmse": math.sqrt(totals["squared_error"] / count),
        "max_abs_error": max_abs,
        "relative_l2": math.sqrt(totals["squared_error"] / totals["ref_squared"]),
        "reference_rms": math.sqrt(totals["ref_squared"] / count),
        "candidate_rms": math.sqrt(totals["other_squared"] / count),
        "global_cosine": totals["dot"]
        / math.sqrt(totals["ref_squared"] * totals["other_squared"]),
        "mean_token_cosine": cosine.mean().item(),
        "min_token_cosine": cosine.min().item(),
        "p01_token_cosine": torch.quantile(cosine, 0.01).item(),
        "median_token_cosine": cosine.median().item(),
        "exact_equal_fraction": exact / count,
        "within_atol_0p01_rtol_0p01_fraction": within / count,
    }
    print(json.dumps(result), flush=True)
    return result, blocks


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(16 * 1024**2), b""):
            digest.update(data)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-rows", type=int, default=4096)
    args = parser.parse_args()
    torch.set_num_threads(8)
    result = {
        "reference": str(args.reference.resolve()),
        "candidate": str(args.candidate.resolve()),
    }
    blocks = []
    layers = []
    with (
        safe_open(args.reference, framework="pt", device="cpu") as reference,
        safe_open(args.candidate, framework="pt", device="cpu") as candidate,
    ):
        assert reference.keys() == candidate.keys()
        assert reference.metadata() == candidate.metadata()
        assert torch.equal(
            reference.get_tensor("input_ids"), candidate.get_tensor("input_ids")
        )
        result["input_ids_identical"] = True
        result["metadata"] = reference.metadata()
        result["tensors"] = {}
        for key in reference.keys():
            a, b = reference.get_slice(key), candidate.get_slice(key)
            assert a.get_shape() == b.get_shape() and a.get_dtype() == b.get_dtype()
            result["tensors"][key] = {"shape": a.get_shape(), "dtype": a.get_dtype()}
        for index, layer_id in enumerate([2, 22, 42, "final_normalized"]):
            key = "target_hidden_states" if index < 3 else "target_last_hidden_states"
            row, layer_blocks = compare_layer(
                reference.get_slice(key),
                candidate.get_slice(key),
                column_start=index * 4096 if index < 3 else 0,
                width=4096,
                label=str(layer_id),
                block_rows=args.block_rows,
            )
            layers.append(row)
            blocks.extend(layer_blocks)
    result["layers"] = layers
    result["files"] = {
        name: {"bytes": path.stat().st_size, "sha256": file_hash(path)}
        for name, path in [("reference", args.reference), ("candidate", args.candidate)]
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    with args.output.with_suffix(".csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(blocks[0]))
        writer.writeheader()
        writer.writerows(blocks)


if __name__ == "__main__":
    main()
