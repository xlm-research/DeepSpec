"""Transactional cache and planning helpers for GLM-5.3 model swapping."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import shutil
from typing import Any

import torch


JOURNAL_FILE_NAME = "partition_journal.json"
JOURNAL_VERSION = 1
CACHE_MARKER = ".deepspec-glm5-partition-cache"
CACHE_MARKER_CONTENT = "deepspec glm5 partition cache v1\n"
MANIFEST_FILE_NAME = "manifest.json"


@dataclass(frozen=True)
class Glm5TrainingPartition:
    partition_id: int
    epoch: int
    start_next_micro_step: int
    end_next_micro_step: int
    optimizer_steps: int
    global_sample_count: int

    def identity(self) -> dict[str, int]:
        return asdict(self)


def compute_glm5_training_partitions(
    *,
    max_samples: int,
    global_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
    max_train_steps: int,
) -> tuple[Glm5TrainingPartition, ...]:
    """Return optimizer-aligned partitions that never cross an epoch boundary."""

    max_samples = int(max_samples)
    global_batch_size = int(global_batch_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)
    micro_batches_per_epoch = int(micro_batches_per_epoch)
    max_train_steps = int(max_train_steps)
    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive.")
    if max_samples < global_batch_size:
        raise ValueError(
            "train.partitioned_model_swap.max_samples must be at least one "
            f"global batch: {max_samples} < {global_batch_size}."
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive.")
    if micro_batches_per_epoch <= 0:
        raise ValueError("micro_batches_per_epoch must be positive.")
    if micro_batches_per_epoch % gradient_accumulation_steps:
        raise ValueError(
            "An epoch must end on an optimizer-step boundary: "
            f"micro_batches_per_epoch={micro_batches_per_epoch}, "
            f"gradient_accumulation_steps={gradient_accumulation_steps}."
        )
    if max_train_steps < 0:
        raise ValueError("max_train_steps must be non-negative.")

    max_steps_per_partition = max_samples // global_batch_size
    total_micro_steps = max_train_steps * gradient_accumulation_steps
    partitions: list[Glm5TrainingPartition] = []
    cursor = 0
    while cursor < total_micro_steps:
        epoch = cursor // micro_batches_per_epoch
        epoch_end = (epoch + 1) * micro_batches_per_epoch
        remaining_micro_steps = min(epoch_end, total_micro_steps) - cursor
        remaining_steps = remaining_micro_steps // gradient_accumulation_steps
        optimizer_steps = min(max_steps_per_partition, remaining_steps)
        if optimizer_steps <= 0:
            raise RuntimeError("Failed to make progress while planning GLM partitions.")
        end = cursor + optimizer_steps * gradient_accumulation_steps
        partitions.append(
            Glm5TrainingPartition(
                partition_id=len(partitions),
                epoch=epoch,
                start_next_micro_step=cursor,
                end_next_micro_step=end,
                optimizer_steps=optimizer_steps,
                global_sample_count=optimizer_steps * global_batch_size,
            )
        )
        cursor = end
    return tuple(partitions)


def atomic_write_json(path: str, value: dict[str, Any]) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)
    _fsync_directory(parent)


def load_json(path: str) -> dict[str, Any] | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(path, flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def validate_glm5_cached_batch(batch: dict[str, torch.Tensor]) -> dict[str, Any]:
    required = (
        "input_ids",
        "loss_mask",
        "target_hidden_states",
        "target_last_hidden_states",
        "context_start",
        "context_len",
        "seq_len",
    )
    missing = [name for name in required if name not in batch]
    if missing:
        raise ValueError(f"GLM partition cache is missing tensors: {missing}.")
    if any(not torch.is_tensor(batch[name]) for name in required):
        raise TypeError("Every required GLM partition-cache value must be a tensor.")

    input_ids = batch["input_ids"]
    loss_mask = batch["loss_mask"]
    selected = batch["target_hidden_states"]
    final = batch["target_last_hidden_states"]
    if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
        raise ValueError(
            "GLM partition cache requires input_ids shaped [1, sequence], got "
            f"{tuple(input_ids.shape)}."
        )
    if tuple(loss_mask.shape) != tuple(input_ids.shape):
        raise ValueError("loss_mask must have the same shape as input_ids.")
    if selected.ndim != 3 or final.ndim != 3:
        raise ValueError("Target hidden states must be rank-three tensors.")
    if int(selected.shape[0]) != 1 or int(final.shape[0]) != 1:
        raise ValueError("Target hidden states must contain exactly one sample.")
    integer_dtypes = {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }
    if input_ids.dtype not in integer_dtypes:
        raise ValueError("input_ids must use an integer dtype.")
    if loss_mask.dtype not in integer_dtypes | {torch.bool}:
        raise ValueError("loss_mask must use a boolean or integer dtype.")
    if not selected.is_floating_point() or not final.is_floating_point():
        raise ValueError("Target hidden states must use floating-point dtypes.")
    if selected.dtype != final.dtype:
        raise ValueError("Target hidden-state tensors must use the same dtype.")
    if int(selected.shape[2]) < 1 or int(final.shape[2]) < 1:
        raise ValueError("Target hidden-state dimensions must be positive.")

    metadata: dict[str, int] = {}
    for name in ("context_start", "context_len", "seq_len"):
        value = batch[name]
        if value.numel() != 1 or value.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError(f"{name} must be one integer tensor value.")
        metadata[name] = int(value.item())
    if metadata["seq_len"] != int(input_ids.shape[1]):
        raise ValueError("seq_len does not match input_ids.")
    if metadata["context_start"] < 0 or metadata["context_len"] < 1:
        raise ValueError("Invalid context_start/context_len in GLM partition cache.")
    if int(selected.shape[1]) != metadata["context_len"]:
        raise ValueError("context_len does not match target_hidden_states.")
    if int(final.shape[1]) != metadata["context_len"]:
        raise ValueError("Target hidden-state tensors have different lengths.")
    if metadata["context_start"] + metadata["context_len"] > metadata["seq_len"]:
        raise ValueError("Cached context range exceeds seq_len.")

    return {
        "token_count": metadata["seq_len"],
        "tensors": {
            name: {
                "shape": list(batch[name].shape),
                "dtype": str(batch[name].dtype).removeprefix("torch."),
            }
            for name in required
        },
    }


class Glm5PartitionCache:
    """One rank's node-local transactional GLM target-feature cache."""

    def __init__(self, *, root: str, global_rank: int):
        self.root = os.path.abspath(os.fspath(root))
        self.global_rank = int(global_rank)
        if os.path.islink(self.root):
            raise ValueError(f"Refusing to use a symlink cache root: {self.root}")
        os.makedirs(self.root, exist_ok=True)
        self.rank_root = os.path.join(self.root, f"rank_{self.global_rank:05d}")
        if os.path.lexists(self.rank_root) and (
            os.path.islink(self.rank_root) or not os.path.isdir(self.rank_root)
        ):
            raise ValueError(f"Invalid GLM rank cache directory: {self.rank_root}")
        os.makedirs(self.rank_root, exist_ok=True)
        marker = os.path.join(self.rank_root, CACHE_MARKER)
        if os.path.exists(marker):
            with open(marker, "r", encoding="utf-8") as handle:
                if handle.read() != CACHE_MARKER_CONTENT:
                    raise ValueError(f"Invalid GLM cache ownership marker: {marker}")
        else:
            with open(marker, "x", encoding="utf-8") as handle:
                handle.write(CACHE_MARKER_CONTENT)
                handle.flush()
                os.fsync(handle.fileno())

    def partition_paths(self, partition: Glm5TrainingPartition) -> tuple[str, str]:
        epoch_dir = os.path.join(self.rank_root, f"epoch_{partition.epoch:04d}")
        stem = f"partition_{partition.partition_id:06d}"
        return os.path.join(epoch_dir, stem + ".incomplete"), os.path.join(
            epoch_dir, stem + ".ready"
        )

    def deleting_path(self, partition: Glm5TrainingPartition) -> str:
        _incomplete, ready = self.partition_paths(partition)
        return ready.removesuffix(".ready") + ".deleting"

    def prepare_incomplete(
        self,
        partition: Glm5TrainingPartition,
        *,
        replace_matching: bool,
    ) -> str:
        incomplete, ready = self.partition_paths(partition)
        if os.path.exists(ready):
            raise FileExistsError(
                f"READY cache already exists and will not be overwritten: {ready}"
            )
        if os.path.lexists(incomplete):
            if os.path.islink(incomplete) or not os.path.isdir(incomplete):
                raise ValueError(f"Invalid incomplete cache path: {incomplete}")
            manifest = load_json(os.path.join(incomplete, MANIFEST_FILE_NAME))
            if manifest is not None:
                self._validate_partition_identity(manifest, partition)
            if not replace_matching:
                raise FileExistsError(f"Incomplete cache already exists: {incomplete}")
            shutil.rmtree(incomplete)
        os.makedirs(incomplete)
        return incomplete

    def write_sample(
        self,
        *,
        partition: Glm5TrainingPartition,
        batch: dict[str, torch.Tensor],
        logical_sample_id: int,
        dataset_index: int,
        stream_micro_step: int,
    ) -> dict[str, Any]:
        incomplete, _ready = self.partition_paths(partition)
        if not os.path.isdir(incomplete):
            raise RuntimeError(f"Incomplete cache is not prepared: {incomplete}")
        validation = validate_glm5_cached_batch(batch)
        file_name = f"sample_{int(logical_sample_id):012d}.pt"
        path = os.path.join(incomplete, file_name)
        if os.path.lexists(path):
            raise FileExistsError(f"Duplicate logical sample cache: {path}")
        tmp_path = f"{path}.tmp-{os.getpid()}"
        cpu_batch = {
            name: value.detach().to(device="cpu") for name, value in batch.items()
        }
        with open(tmp_path, "xb") as handle:
            torch.save(cpu_batch, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        _fsync_directory(incomplete)
        batch.clear()
        return {
            "file": file_name,
            "writer_rank": self.global_rank,
            "logical_sample_id": int(logical_sample_id),
            "dataset_index": int(dataset_index),
            "stream_micro_step": int(stream_micro_step),
            "token_count": int(validation["token_count"]),
            "file_size": int(os.path.getsize(path)),
            "tensors": validation["tensors"],
        }

    def write_local_manifest(
        self,
        *,
        partition: Glm5TrainingPartition,
        samples: list[dict[str, Any]],
        target_shard_layout: dict[str, Any],
        state: str,
    ) -> dict[str, Any]:
        incomplete, _ready = self.partition_paths(partition)
        manifest = {
            "version": 1,
            "state": str(state),
            **partition.identity(),
            "writer_rank": self.global_rank,
            "training_stream": {
                "start_next_micro_step": partition.start_next_micro_step,
                "end_next_micro_step": partition.end_next_micro_step,
            },
            "local_sample_count": len(samples),
            "local_token_count": sum(int(item["token_count"]) for item in samples),
            "local_file_size": sum(int(item["file_size"]) for item in samples),
            "target_shard_layout": target_shard_layout,
            "samples": samples,
        }
        atomic_write_json(os.path.join(incomplete, MANIFEST_FILE_NAME), manifest)
        return manifest

    def commit_ready(self, partition: Glm5TrainingPartition) -> str:
        incomplete, ready = self.partition_paths(partition)
        manifest = load_json(os.path.join(incomplete, MANIFEST_FILE_NAME))
        if manifest is None:
            raise FileNotFoundError(f"Local cache manifest is missing: {incomplete}")
        self._validate_partition_identity(manifest, partition)
        manifest["state"] = "READY"
        atomic_write_json(os.path.join(incomplete, MANIFEST_FILE_NAME), manifest)
        os.rename(incomplete, ready)
        _fsync_directory(os.path.dirname(ready))
        return ready

    def validate_incomplete(
        self, partition: Glm5TrainingPartition
    ) -> tuple[str, dict[str, Any]]:
        incomplete, _ready = self.partition_paths(partition)
        return self._validate_manifest_directory(
            incomplete,
            partition=partition,
            expected_state="LOCAL_COMPLETE",
        )

    def validate_ready(
        self, partition: Glm5TrainingPartition
    ) -> tuple[str, dict[str, Any]]:
        _incomplete, ready = self.partition_paths(partition)
        return self._validate_manifest_directory(
            ready,
            partition=partition,
            expected_state="READY",
        )

    def _validate_manifest_directory(
        self,
        directory: str,
        *,
        partition: Glm5TrainingPartition,
        expected_state: str,
    ) -> tuple[str, dict[str, Any]]:
        if os.path.islink(directory) or not os.path.isdir(directory):
            raise FileNotFoundError(f"Partition cache does not exist: {directory}")
        manifest = load_json(os.path.join(directory, MANIFEST_FILE_NAME))
        if manifest is None or manifest.get("state") != expected_state:
            raise ValueError(
                f"Partition cache has no {expected_state} manifest: {directory}"
            )
        self._validate_partition_identity(manifest, partition)
        samples = manifest.get("samples")
        if not isinstance(samples, list) or len(samples) != int(
            manifest.get("local_sample_count", -1)
        ):
            raise ValueError(f"Invalid sample list in {directory}.")
        if int(manifest.get("writer_rank", -1)) != self.global_rank:
            raise ValueError(f"Wrong manifest writer rank in {directory}.")
        if not isinstance(manifest.get("target_shard_layout"), dict):
            raise ValueError(f"Missing target shard layout in {directory}.")
        seen_ids: set[int] = set()
        seen_files: set[str] = set()
        token_count = 0
        file_size = 0
        for sample in samples:
            if int(sample.get("writer_rank", -1)) != self.global_rank:
                raise ValueError(f"Wrong writer rank in {directory}.")
            logical_id = int(sample["logical_sample_id"])
            if logical_id in seen_ids:
                raise ValueError(
                    f"Duplicate logical sample {logical_id} in {directory}."
                )
            seen_ids.add(logical_id)
            file_name = sample.get("file")
            if (
                not isinstance(file_name, str)
                or os.path.basename(file_name) != file_name
                or file_name in seen_files
            ):
                raise ValueError(f"Invalid or duplicate sample file in {directory}.")
            seen_files.add(file_name)
            if file_name != f"sample_{logical_id:012d}.pt":
                raise ValueError(f"Sample file identity changed in {directory}.")
            stream_micro_step = int(sample.get("stream_micro_step", -1))
            if not (
                partition.start_next_micro_step
                <= stream_micro_step
                < partition.end_next_micro_step
            ):
                raise ValueError(f"Sample stream position is outside {directory}.")
            if int(sample.get("dataset_index", -1)) < 0:
                raise ValueError(f"Invalid dataset index in {directory}.")
            path = os.path.join(directory, file_name)
            actual_file_size = int(os.path.getsize(path))
            if actual_file_size != int(sample["file_size"]):
                raise ValueError(f"Cached sample size changed: {path}")
            batch = torch.load(path, map_location="cpu", weights_only=True)
            validation = validate_glm5_cached_batch(batch)
            if validation["tensors"] != sample["tensors"]:
                raise ValueError(f"Cached tensor metadata changed: {path}")
            if int(sample.get("token_count", -1)) != int(
                validation["token_count"]
            ):
                raise ValueError(f"Cached token count changed: {path}")
            token_count += int(validation["token_count"])
            file_size += actual_file_size
        if token_count != int(manifest.get("local_token_count", -1)):
            raise ValueError(f"Manifest token count changed in {directory}.")
        if file_size != int(manifest.get("local_file_size", -1)):
            raise ValueError(f"Manifest file size changed in {directory}.")
        return directory, manifest

    def delete_ready(self, partition: Glm5TrainingPartition) -> None:
        self.begin_delete(partition)
        self.finish_delete(partition)

    def begin_delete(self, partition: Glm5TrainingPartition) -> str:
        _incomplete, ready = self.partition_paths(partition)
        deleting = self.deleting_path(partition)
        if os.path.isdir(ready):
            self.validate_ready(partition)
            os.rename(ready, deleting)
            _fsync_directory(os.path.dirname(deleting))
            return "renamed"
        if os.path.isdir(deleting):
            manifest = load_json(os.path.join(deleting, MANIFEST_FILE_NAME))
            if manifest is None or manifest.get("state") != "READY":
                raise ValueError(f"Invalid deleting cache: {deleting}")
            self._validate_partition_identity(manifest, partition)
            return "already_renamed"
        if os.path.lexists(ready) or os.path.lexists(deleting):
            raise ValueError("Partition cache deletion path is not a directory.")
        return "already_deleted"

    def finish_delete(self, partition: Glm5TrainingPartition) -> None:
        _incomplete, ready = self.partition_paths(partition)
        deleting = self.deleting_path(partition)
        if os.path.exists(ready):
            raise RuntimeError("Cache must be renamed before deletion.")
        if not os.path.exists(deleting):
            return
        manifest = load_json(os.path.join(deleting, MANIFEST_FILE_NAME))
        if manifest is None:
            raise FileNotFoundError(
                f"Refusing to delete cache without manifest: {deleting}"
            )
        self._validate_partition_identity(manifest, partition)
        expected_parent = os.path.realpath(os.path.dirname(deleting))
        if os.path.dirname(expected_parent) != os.path.realpath(self.rank_root):
            raise RuntimeError(f"Cache path escaped rank root: {deleting}")
        shutil.rmtree(deleting)
        try:
            os.rmdir(os.path.dirname(deleting))
        except OSError:
            pass

    @staticmethod
    def _validate_partition_identity(
        value: dict[str, Any], partition: Glm5TrainingPartition
    ) -> None:
        for name, expected in partition.identity().items():
            if int(value.get(name, -1)) != int(expected):
                raise ValueError(
                    f"Partition cache identity mismatch for {name}: "
                    f"{value.get(name)!r} != {expected!r}."
                )


class Glm5ReadyCacheLoader:
    """Finite CPU loader for one validated local READY manifest."""

    def __init__(self, *, ready_dir: str, manifest: dict[str, Any]):
        self.ready_dir = os.path.abspath(ready_dir)
        self.samples = tuple(manifest["samples"])

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        for sample in self.samples:
            path = os.path.join(self.ready_dir, os.path.basename(sample["file"]))
            yield torch.load(path, map_location="cpu", weights_only=True)

    def close(self) -> None:
        return None


def journal_path(checkpoint_dir_root: str) -> str:
    return os.path.join(os.path.abspath(checkpoint_dir_root), JOURNAL_FILE_NAME)


def build_journal_record(
    *,
    phase: str,
    partition: Glm5TrainingPartition,
    run_identity: dict[str, Any],
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    return {
        "version": JOURNAL_VERSION,
        "mode": "glm5_next_partitioned_model_swap",
        "phase": str(phase),
        "partition": partition.identity(),
        "run_identity": run_identity,
        "checkpoint_dir": checkpoint_dir,
    }


def validate_journal_record(
    value: dict[str, Any],
    *,
    partition: Glm5TrainingPartition,
    run_identity: dict[str, Any],
) -> None:
    if int(value.get("version", -1)) != JOURNAL_VERSION:
        raise ValueError("Unsupported GLM partition journal version.")
    if value.get("mode") != "glm5_next_partitioned_model_swap":
        raise ValueError("Partition journal belongs to another training mode.")
    saved_identity = value.get("run_identity")
    if saved_identity != run_identity:
        raise ValueError(
            "Partition journal run identity does not match the current training run."
        )
    saved_partition = value.get("partition")
    if not isinstance(saved_partition, dict):
        raise ValueError("Partition journal has no partition identity.")
    Glm5PartitionCache._validate_partition_identity(saved_partition, partition)


__all__ = [
    "Glm5PartitionCache",
    "Glm5ReadyCacheLoader",
    "Glm5TrainingPartition",
    "build_journal_record",
    "compute_glm5_training_partitions",
    "journal_path",
    "load_json",
    "atomic_write_json",
    "validate_glm5_cached_batch",
    "validate_journal_record",
]
