"""Validate immutable producer facts and reconstruct original token order."""

import hashlib
import json
from pathlib import Path

import torch

from .planning import input_identity


def file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def context_positions(length, size, rank):
    if size == 1:
        return torch.arange(length)
    chunk = (length + 2 * size - 1) // (2 * size)
    offsets = torch.arange(chunk)
    return torch.cat((offsets + rank * chunk, offsets + (2 * size - rank - 1) * chunk))


class ProducerFeatures:
    def __init__(self, path, digest, *, layer_ids, hidden_size, vocab_size):
        self.root = Path(path).resolve().parent
        raw = Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("Producer manifest has changed")
        manifest = json.loads(raw)
        if manifest["version"] != 1:
            raise ValueError("Unsupported producer feature version")
        teacher = manifest["teacher"]
        for name, expected in (
            ("target_layer_ids", layer_ids),
            ("hidden_size", hidden_size),
            ("activation_dtype", "bfloat16"),
            ("target_final_hidden_source", "full_model_final_norm_output"),
        ):
            if teacher[name] != expected:
                raise ValueError(f"Producer {name} differs from the native recipe")
        self.hidden_size = hidden_size
        self.num_layers = len(layer_ids)
        self.vocab_size = vocab_size
        self.samples = {sample["sample_id"]: sample for sample in manifest["samples"]}
        if len(self.samples) != len(manifest["samples"]):
            raise ValueError("Duplicate producer sample identity")

    def read(self, sample_id, expected_input):
        sample = self.samples[sample_id]
        if sample["input_identity"] != expected_input:
            raise ValueError("Producer sample differs from the native input plan")
        length = sample["length"]
        shards = sample["shards"]
        if (
            length < 1
            or not shards
            or [s["cp_rank"] for s in shards] != list(range(len(shards)))
        ):
            raise ValueError("Producer sequence or context shards are incomplete")
        full = None
        for shard in shards:
            path = self.root / shard["path"]
            if file_digest(path) != shard["sha256"]:
                raise ValueError("Producer feature bytes have changed")
            batch = torch.load(path, weights_only=True, map_location="cpu")
            tokens = batch["input_ids"]
            mask = batch["loss_mask"]
            if (
                tokens.shape != (1, length)
                or tokens.dtype != torch.int64
                or mask.shape != tokens.shape
                or mask.dtype not in (torch.bool, torch.int64, torch.float32)
                or not torch.isfinite(mask).all()
                or (mask < 0).any()
                or tokens.min() < 0
                or tokens.max() >= self.vocab_size
                or batch["seq_len"].numel() != 1
                or batch["seq_len"].item() != length
                or input_identity(batch) != expected_input
            ):
                raise ValueError("Producer tokens, mask or sequence length are invalid")
            positions = context_positions(length, len(shards), shard["cp_rank"])
            if batch["context_chunk_len"].numel() != 1 or batch[
                "context_chunk_len"
            ].item() != len(positions):
                raise ValueError("Producer context length differs from its shard")
            if full is None:
                full = {
                    key: batch[key] for key in ("input_ids", "loss_mask", "seq_len")
                }
                full["context_chunk_len"] = torch.tensor([length])
            valid = positions < length
            for key, width in (
                ("target_hidden_states", self.hidden_size * self.num_layers),
                ("target_last_hidden_states", self.hidden_size),
            ):
                tensor = batch[key]
                if (
                    tensor.shape != (1, len(positions), width)
                    or tensor.dtype != torch.bfloat16
                    or not torch.isfinite(tensor).all()
                ):
                    raise ValueError(
                        "Producer hidden shape, dtype or values are invalid"
                    )
                if key not in full:
                    full[key] = tensor.new_empty((1, length, width))
                full[key][:, positions[valid]] = tensor[:, valid]
        return full
