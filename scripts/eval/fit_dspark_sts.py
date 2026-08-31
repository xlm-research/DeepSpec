#!/usr/bin/env python3
"""Fit DSpark Sequential Temperature Scaling from evaluation observations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from deepspec.eval.dspark.scheduler import (
    DEFAULT_TEMPERATURE_GRID,
    SequentialTemperatureScaler,
    expected_calibration_error,
)


def load_observations(
    paths: list[str | Path],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str], dict]:
    rows: list[tuple[list[float], list[int]]] = []
    sources = []
    shared_metadata = None
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        sources.append(str(path))
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"Could not read observation file {path}: {exc}") from exc
        with handle:
            source_metadata = None
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if payload.get("record_type") == "metadata":
                    if source_metadata is not None:
                        raise ValueError(
                            f"{path}:{line_number}: duplicate metadata."
                        )
                    source_metadata = {
                        key: payload.get(key)
                        for key in (
                            "schema_version",
                            "target_model",
                            "draft_model",
                            "sampling",
                            "block_size",
                        )
                    }
                    continue
                if source_metadata is None:
                    raise ValueError(
                        f"{path}:{line_number}: observation metadata must be first."
                    )
                if payload.get("record_type") != "observation":
                    raise ValueError(
                        f"{path}:{line_number}: unknown record_type."
                    )
                logits = payload.get("confidence_logits")
                targets = payload.get("prefix_targets")
                if (
                    not isinstance(logits, list)
                    or not isinstance(targets, list)
                    or not logits
                    or len(logits) != len(targets)
                ):
                    raise ValueError(
                        f"{path}:{line_number}: confidence_logits and "
                        "prefix_targets must be equal-length non-empty lists."
                    )
                rows.append((logits, targets))
            if source_metadata is None:
                raise ValueError(f"{path}: missing observation metadata.")
            if int(source_metadata.get("schema_version") or -1) != 1:
                raise ValueError(f"{path}: unsupported observation schema.")
            if not all(
                isinstance(source_metadata.get(name), str)
                and bool(source_metadata[name])
                for name in ("target_model", "draft_model")
            ):
                raise ValueError(f"{path}: missing target/draft model metadata.")
            sampling = source_metadata.get("sampling")
            if not isinstance(sampling, dict):
                raise ValueError(f"{path}: missing sampling metadata.")
            try:
                sampling_temperature = float(sampling["temperature"])
                sampling_top_k = int(sampling["top_k"])
                sampling_top_p = float(sampling["top_p"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{path}: invalid sampling metadata.") from exc
            if (
                sampling_temperature < 0.0
                or sampling_top_k < 0
                or not 0.0 < sampling_top_p <= 1.0
            ):
                raise ValueError(f"{path}: invalid sampling metadata.")
            source_metadata["sampling"] = {
                "temperature": sampling_temperature,
                "top_k": sampling_top_k,
                "top_p": sampling_top_p,
            }
            if shared_metadata is None:
                shared_metadata = source_metadata
            elif source_metadata != shared_metadata:
                raise ValueError(
                    "Observation files use different model or sampling metadata."
                )
    if not rows:
        raise ValueError("No confidence observations were found.")

    block_size = max(len(logits) for logits, _ in rows)
    confidence_logits = torch.zeros(len(rows), block_size, dtype=torch.float32)
    prefix_targets = torch.zeros(len(rows), block_size, dtype=torch.float32)
    valid_mask = torch.zeros(len(rows), block_size, dtype=torch.bool)
    for row_idx, (logits, targets) in enumerate(rows):
        length = len(logits)
        try:
            confidence_logits[row_idx, :length] = torch.tensor(
                logits,
                dtype=torch.float32,
            )
            prefix_targets[row_idx, :length] = torch.tensor(
                targets,
                dtype=torch.float32,
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(f"Observation row {row_idx} is not numeric.") from exc
        valid_mask[row_idx, :length] = True
    assert shared_metadata is not None
    if int(shared_metadata.get("block_size") or -1) != block_size:
        raise ValueError(
            "Observation metadata block_size does not match observed rows."
        )
    return confidence_logits, prefix_targets, valid_mask, sources, shared_metadata


def build_calibration_artifact(
    *,
    scaler: SequentialTemperatureScaler,
    confidence_logits: torch.Tensor,
    prefix_targets: torch.Tensor,
    valid_mask: torch.Tensor,
    target_model: str,
    draft_model: str,
    temperature: float,
    top_k: int,
    top_p: float,
    sources: list[str],
) -> dict:
    raw_prefix = confidence_logits.sigmoid().cumprod(dim=-1)
    calibrated_prefix = scaler.calibrate_logits(confidence_logits).cumprod(dim=-1)
    raw_ece = []
    calibrated_ece = []
    for position in range(scaler.block_size):
        position_mask = valid_mask[:, position]
        raw_ece.append(float(expected_calibration_error(
            raw_prefix[:, position],
            prefix_targets[:, position],
            num_bins=scaler.num_bins,
            valid_mask=position_mask,
        ).item()))
        calibrated_ece.append(float(expected_calibration_error(
            calibrated_prefix[:, position],
            prefix_targets[:, position],
            num_bins=scaler.num_bins,
            valid_mask=position_mask,
        ).item()))
    return {
        "schema_version": 1,
        "method": "sequential_temperature_scaling",
        "target_model": str(target_model),
        "draft_model": str(draft_model),
        "sampling": {
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
        },
        "block_size": scaler.block_size,
        "temperatures": scaler.temperatures.tolist(),
        "num_bins": scaler.num_bins,
        "num_observations": int(confidence_logits.shape[0]),
        "sources": sources,
        "raw_prefix_ece": raw_ece,
        "calibrated_prefix_ece": calibrated_ece,
    }


def write_json_atomic(path: str | Path, payload: dict) -> Path:
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f"{output_path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, output_path)
    return output_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations-jsonl", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-bins", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    (
        confidence_logits,
        prefix_targets,
        valid_mask,
        sources,
        metadata,
    ) = load_observations(args.observations_jsonl)
    scaler = SequentialTemperatureScaler.fit(
        confidence_logits,
        prefix_targets,
        temperature_grid=DEFAULT_TEMPERATURE_GRID,
        num_bins=args.num_bins,
        valid_mask=valid_mask,
    )
    payload = build_calibration_artifact(
        scaler=scaler,
        confidence_logits=confidence_logits,
        prefix_targets=prefix_targets,
        valid_mask=valid_mask,
        target_model=metadata["target_model"],
        draft_model=metadata["draft_model"],
        temperature=metadata["sampling"]["temperature"],
        top_k=metadata["sampling"]["top_k"],
        top_p=metadata["sampling"]["top_p"],
        sources=sources,
    )
    output_path = write_json_atomic(args.output, payload)
    print(f"Wrote STS calibration to {output_path}", flush=True)


if __name__ == "__main__":
    main()
