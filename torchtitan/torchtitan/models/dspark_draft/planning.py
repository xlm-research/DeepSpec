"""Immutable input identity shared by CPU preparation and feature consumption."""

import hashlib
import json
from pathlib import Path

import torch


def input_identity(batch):
    digest = hashlib.sha256()
    for key, dtype in (("input_ids", torch.int64), ("loss_mask", torch.float32)):
        value = batch[key].detach().to(device="cpu", dtype=dtype).contiguous()
        digest.update(json.dumps(list(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def make_input_plan(*, run_id, ordered_batches):
    return {
        "version": 1,
        "run_id": run_id,
        "batches": [
            {"id": batch_id, "input_identity": input_identity(batch)}
            for batch_id, batch in ordered_batches
        ],
    }


def model_identity(path):
    root = Path(path).resolve()
    index = json.loads((root / "model.safetensors.index.json").read_text())
    weights = []
    for name in sorted(set(index["weight_map"].values())):
        stat = (root / name).stat()
        weights.append((name, stat.st_size, stat.st_mtime_ns))
    return {
        "model_path": str(root),
        "config_sha256": hashlib.sha256(
            (root / "config.json").read_bytes()
        ).hexdigest(),
        "index_sha256": hashlib.sha256(
            (root / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "weights_sha256": hashlib.sha256(json.dumps(weights).encode()).hexdigest(),
    }
