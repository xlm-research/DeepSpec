"""Index immutable target files independently of the draft's CP/TP layout."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import torch


def _file_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def feature_input_identity(batch):
    """Bind a producer output to the requested tokens and supervision mask."""
    digest = hashlib.sha256()
    for name, dtype in (("input_ids", torch.int64), ("loss_mask", torch.float32)):
        tensor = batch[name].detach().to(device="cpu", dtype=dtype).contiguous()
        digest.update(json.dumps(list(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _context_positions(length, size, rank):
    if size < 1 or not 0 <= rank < size:
        raise ValueError("Invalid context-parallel coordinate.")
    if size == 1:
        return torch.arange(length)
    chunk = (length + 2 * size - 1) // (2 * size)
    offsets = torch.arange(chunk)
    return torch.cat((offsets + rank * chunk, offsets + (2 * size - rank - 1) * chunk))


class DraftFeatureIndex:
    """One phase of ordered samples, with one sample per draft DP rank/microbatch.

    Source file owners and head/tail shards belong to the fixed producer. Only
    reading constructs the consumer's context view; TP peers read the same DP
    sample. The index is durable metadata and never rewrites producer tensors.
    """

    def __init__(self, manifest):
        self._manifest = copy.deepcopy(manifest)
        self.start_micro_step = int(manifest["start_micro_step"])
        self.data_parallel_size = int(manifest["data_parallel_size"])
        self.gradient_accumulation_steps = int(manifest["gradient_accumulation_steps"])
        self._samples = self._manifest["samples"]
        self._producer_cp_size = int(
            manifest.get("producer_identity", {}).get("layout", {}).get("cp", 0)
        )
        if (
            manifest["version"] != 1
            or self.start_micro_step < 0
            or self.data_parallel_size < 1
            or self._producer_cp_size < 1
            or self.gradient_accumulation_steps < 1
            or int(manifest["samples_per_epoch"]) < 1
            or not self._samples
            or len(self._samples) % self.data_parallel_size
        ):
            raise ValueError("Invalid draft feature index schedule.")
        start_position = self.start_micro_step * self.data_parallel_size
        seen = set()
        seen_paths = set()
        for offset, sample in enumerate(self._samples):
            position = start_position + offset
            micro_step = position // self.data_parallel_size
            if (
                sample["position"] != position
                or sample["micro_step"] != micro_step
                or sample["update_step"]
                != micro_step // self.gradient_accumulation_steps
                or sample["epoch"] != position // int(manifest["samples_per_epoch"])
                or sample["sample_id"] in seen
            ):
                raise ValueError(
                    "Draft sample identity, order, or update membership mismatch."
                )
            seen.add(sample["sample_id"])
            shards = sample["shards"]
            if not shards or [shard["cp_rank"] for shard in shards] != list(
                range(self._producer_cp_size)
            ):
                raise ValueError("Missing or duplicate producer context shards.")
            for shard in shards:
                path = os.path.realpath(shard["path"])
                if path in seen_paths:
                    raise ValueError(
                        "A producer feature file is reused for another shard."
                    )
                seen_paths.add(path)
        self.end_micro_step = (
            self.start_micro_step + len(self._samples) // self.data_parallel_size
        )

    @classmethod
    def create(
        cls,
        *,
        samples,
        producer_identity,
        partition_id,
        start_micro_step,
        data_parallel_size,
        gradient_accumulation_steps,
        samples_per_epoch,
    ):
        records = copy.deepcopy(sorted(samples, key=lambda sample: sample["position"]))
        for sample in records:
            sample["micro_step"] = sample["position"] // data_parallel_size
            sample["update_step"] = sample["micro_step"] // gradient_accumulation_steps
            sample["shards"].sort(key=lambda shard: shard["cp_rank"])
            for shard in sample["shards"]:
                shard["path"] = str(Path(shard["path"]).resolve())
                shard["sha256"] = _file_digest(shard["path"])
        return cls(
            {
                "version": 1,
                "producer_identity": producer_identity,
                "partition_id": partition_id,
                "start_micro_step": start_micro_step,
                "data_parallel_size": data_parallel_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "samples_per_epoch": samples_per_epoch,
                "samples": records,
            }
        )

    @property
    def identity(self):
        encoded = json.dumps(self._manifest, sort_keys=True).encode()
        return hashlib.sha256(encoded).hexdigest()

    def save(self, path):
        path = Path(path)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        with temporary.open("x") as stream:
            json.dump(self._manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    @classmethod
    def load(
        cls,
        path,
        *,
        producer_identity,
        next_micro_step,
        data_parallel_size,
        gradient_accumulation_steps,
    ):
        manifest = json.loads(Path(path).read_text())
        for name, expected in (
            ("producer_identity", producer_identity),
            ("data_parallel_size", data_parallel_size),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
        ):
            if manifest.get(name) != expected:
                raise ValueError(f"Draft feature index {name} mismatch.")
        index = cls(manifest)
        if not index.start_micro_step <= next_micro_step <= index.end_micro_step:
            raise ValueError("Resume cursor is outside the indexed draft phase.")
        return index

    def owned_paths(self, rank):
        return tuple(
            shard["path"]
            for sample in self._samples
            for shard in sample["shards"]
            if shard["owner"] == rank
        )

    def read(
        self,
        *,
        micro_step,
        data_parallel_rank,
        context_parallel_size=1,
        context_parallel_rank=0,
    ):
        if not (
            self.start_micro_step <= micro_step < self.end_micro_step
            and 0 <= data_parallel_rank < self.data_parallel_size
        ):
            raise ValueError("Microbatch is outside the indexed draft phase.")
        offset = (
            micro_step - self.start_micro_step
        ) * self.data_parallel_size + data_parallel_rank
        sample = self._samples[offset]
        full = None
        for shard in sample["shards"]:
            if _file_digest(shard["path"]) != shard["sha256"]:
                raise ValueError(f"Producer feature identity changed: {shard['path']}")
            batch = torch.load(shard["path"], map_location="cpu", weights_only=True)
            if (
                "input_identity" in sample
                and feature_input_identity(batch) != sample["input_identity"]
            ):
                raise ValueError(
                    "Producer feature input identity does not match its indexed sample."
                )
            length = int(batch["seq_len"].item())
            positions = _context_positions(
                length, self._producer_cp_size, shard["cp_rank"]
            )
            if full is None:
                if batch["input_ids"].shape != (1, length):
                    raise ValueError(
                        "Producer tokens do not match the indexed sequence length."
                    )
                full = {
                    name: batch[name] for name in ("input_ids", "loss_mask", "seq_len")
                }
                for name in ("target_hidden_states", "target_last_hidden_states"):
                    full[name] = batch[name].new_zeros(
                        (1, length, batch[name].shape[-1])
                    )
            for name in ("input_ids", "loss_mask", "seq_len"):
                if not torch.equal(batch[name], full[name]):
                    raise ValueError("Producer shards belong to different samples.")
            valid = positions < length
            for name in ("target_hidden_states", "target_last_hidden_states"):
                if batch[name].shape != (1, len(positions), full[name].shape[-1]):
                    raise ValueError("Incomplete producer feature shard.")
                full[name][:, positions[valid]] = batch[name][:, valid]
        assert full is not None
        positions = _context_positions(
            length, context_parallel_size, context_parallel_rank
        )
        for name in ("target_hidden_states", "target_last_hidden_states"):
            selected = full[name][:, positions.clamp_max(length - 1)].contiguous()
            selected[:, positions >= length] = 0
            full[name] = selected
        full["context_chunk_len"] = torch.tensor([len(positions)])
        return full
