"""Synchronous, independently durable checkpoints for completed draft phases."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch.distributed as dist
from torch.distributed.checkpoint import CheckpointException, FileSystemReader

from deepspec.data.draft_feature_reader import DraftFeatureIndex
from deepspec.distributed.distributed_checkpoint import (
    read_checkpoint_metadata,
    save_training_checkpoint,
    write_checkpoint_metadata,
)
from deepspec.trainer.ckpt_manager import save_train_config
from deepspec.trainer.glm5_partitioned_swap import atomic_write_json
from deepspec.utils.config import CustomJSONEncoder


_COMMIT_FILE = "draft_phase_commit.json"


def _fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _root_action(action, control_group):
    result: list[object] = [None, None]
    if dist.get_rank() == 0:
        try:
            result[0] = action()
        except Exception as exc:
            result[1] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(result, src=0, group=control_group)
    if result[1] is not None:
        raise RuntimeError(f"Draft phase checkpoint failed: {result[1]}")
    return result[0]


def validate_draft_phase_checkpoint(path):
    """Reject a missing, truncated or changed part of a committed full state."""
    path = Path(path)
    commit = json.loads((path / _COMMIT_FILE).read_text())
    if commit.get("version") != 1 or not commit.get("files"):
        raise ValueError("Invalid draft phase checkpoint commit record.")
    for name, expected in commit["files"].items():
        file = path / name
        if Path(name).is_absolute() or ".." in Path(name).parts or file.is_symlink():
            raise ValueError("Invalid draft checkpoint file path.")
        if (
            file.stat().st_size != expected["size"]
            or _digest(file) != expected["sha256"]
        ):
            raise ValueError(f"Incomplete or changed draft checkpoint file: {name}")
    required = {
        "train_config.py",
        "resolved_train_config.json",
        "draft_feature_index.json",
        "distributed_checkpoint_metadata.json",
        "distributed_checkpoint/.metadata",
    }
    if not required.issubset(commit["files"]):
        raise ValueError("Draft checkpoint lacks required configuration or metadata.")
    metadata = read_checkpoint_metadata(str(path))
    storage_files = _verify_distributed_files(path, metadata["saved_world_size"])
    if not storage_files.issubset(commit["files"]):
        raise ValueError("Draft checkpoint storage is missing from its commit record.")
    manifest = json.loads((path / "draft_feature_index.json").read_text())
    index = DraftFeatureIndex(manifest)
    if commit["feature_index"] != index.identity:
        raise ValueError("Draft checkpoint input index identity mismatch.")
    resolved = json.loads((path / "resolved_train_config.json").read_text())
    parallel = metadata["parallel_config"]
    dp_size = parallel["dp_replicate"] * parallel["dp_shard"]
    world_size = dp_size * parallel["cp"] * parallel["tp"] * parallel["pp"]
    local_batch = resolved["train"]["local_batch_size"]
    gas = index.gradient_accumulation_steps
    if (
        world_size != metadata["saved_world_size"]
        or dp_size != index.data_parallel_size
        or local_batch != 1
        or resolved["train"]["global_batch_size"] != dp_size * local_batch * gas
    ):
        raise ValueError("Draft checkpoint topology or accumulation mismatch.")
    expected_progress = {
        "next_micro_step": index.end_micro_step,
        "global_step": index.end_micro_step // gas,
        "partition_id": index.partition_id,
        "partition_start_next_micro_step": index.start_micro_step,
        "partition_end_next_micro_step": index.end_micro_step,
        "epoch": index.end_micro_step * dp_size // manifest["samples_per_epoch"],
        "data_position": index.end_micro_step * local_batch,
        "checkpointed": True,
    }
    if (
        index.start_micro_step % gas
        or index.end_micro_step % gas
        or any(metadata.get(name) != value for name, value in expected_progress.items())
        or commit["next_micro_step"] != metadata["next_micro_step"]
        or commit["global_step"] != metadata["global_step"]
    ):
        raise ValueError("Draft checkpoint progress disagrees with its phase index.")
    suffix = path.name.removeprefix("step_")
    if (
        path.name.startswith("step_")
        and suffix.isdigit()
        and int(suffix) != commit["global_step"]
    ):
        raise ValueError("Checkpoint directory and progress mismatch.")
    return commit


def _verify_distributed_files(path, world_size):
    metadata = FileSystemReader(path / "distributed_checkpoint").read_metadata()
    keys = metadata.state_dict_metadata
    for prefix in ("model.", "optimizer.state.", "scheduler."):
        if not any(key.startswith(prefix) for key in keys):
            raise ValueError(f"Draft checkpoint lacks {prefix} state.")
    for rank in range(world_size):
        for name in (
            "torch_rng",
            "torch_cuda_rng",
            "numpy_rng_pickle",
            "python_rng_pickle",
        ):
            if f"training_rank_{rank}.{name}" not in keys:
                raise ValueError(f"Draft checkpoint lacks rank {rank} RNG state.")
    files = set()
    for storage in metadata.storage_data.values():
        if (
            Path(storage.relative_path).is_absolute()
            or ".." in Path(storage.relative_path).parts
        ):
            raise ValueError("Invalid distributed checkpoint storage path.")
        file = path / "distributed_checkpoint" / storage.relative_path
        if file.stat().st_size < storage.offset + storage.length:
            raise ValueError(f"Incomplete distributed checkpoint storage: {file}")
        files.add(str(file.relative_to(path)))
    return files


def discover_draft_phase_checkpoint(root, *, control_group=None):
    """Select the newest valid committed state, ignoring incomplete attempts."""

    def discover():
        candidates = [
            path
            for path in Path(root).glob("step_*")
            if path.name.removeprefix("step_").isdigit()
            and not path.is_symlink()
            and (path / _COMMIT_FILE).is_file()
        ]
        failures = []
        for path in sorted(candidates, key=lambda p: int(p.name[5:]), reverse=True):
            try:
                validate_draft_phase_checkpoint(path)
                return str(path.resolve())
            except Exception as exc:
                failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
                print(
                    f"[draft-checkpoint] ignoring invalid checkpoint: {failures[-1]}",
                    flush=True,
                )
        if failures:
            raise ValueError(
                "No valid committed draft checkpoint: " + "; ".join(failures)
            )
        return None

    return _root_action(discover, control_group)


def validate_draft_phase_resume(path, *, train_config, progress, control_group=None):
    """Check current settings against an already validated committed checkpoint."""

    def validate():
        checkpoint = Path(path)
        saved = json.loads((checkpoint / "resolved_train_config.json").read_text())
        current = json.loads(json.dumps(train_config, cls=CustomJSONEncoder))
        for section in ("model", "train"):
            if saved[section] != current[section]:
                raise ValueError(f"Draft resume {section} configuration mismatch.")
        metadata = read_checkpoint_metadata(str(checkpoint))
        for name in ("saved_world_size", "parallel_config", "model_config"):
            expected = json.loads(
                json.dumps(getattr(progress, name), cls=CustomJSONEncoder)
            )
            if metadata[name] != expected:
                raise ValueError(f"Draft resume {name} mismatch.")
        return metadata

    return _root_action(validate, control_group)


def save_draft_phase_checkpoint(
    *,
    checkpoint_dir_root,
    model,
    optimizer_bundle,
    progress,
    train_config,
    feature_index,
    control_group,
):
    """Publish a phase only after DCP, its input index and metadata are durable."""
    if progress.next_micro_step != feature_index.end_micro_step:
        raise ValueError("A phase checkpoint must follow its final microbatch.")
    root = Path(checkpoint_dir_root)
    destination = root / f"step_{progress.global_step}"

    def prepare():
        root.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                validate_draft_phase_checkpoint(destination)
            except Exception:
                # A restart may replay a damaged or never-committed step.
                # Preserve its evidence outside the automatic discovery set.
                rejected = Path(
                    tempfile.mkdtemp(
                        prefix=f".step_{progress.global_step}.rejected-", dir=root
                    )
                )
                os.rename(destination, rejected / "checkpoint")
                _fsync_directory(rejected)
                _fsync_directory(root)
            else:
                raise FileExistsError(
                    f"A draft checkpoint is already committed: {destination}"
                )
        return tempfile.mkdtemp(
            prefix=f".step_{progress.global_step}.incomplete-", dir=root
        )

    temporary = _root_action(prepare, control_group)
    write_dir = Path(temporary)

    def save_configuration():
        save_train_config(train_config=train_config, checkpoint_dir=str(write_dir))
        resolved = json.loads(json.dumps(train_config, cls=CustomJSONEncoder))
        atomic_write_json(write_dir / "resolved_train_config.json", resolved)
        feature_index.save(write_dir / "draft_feature_index.json")

    _root_action(save_configuration, control_group)
    # PyTorch DCP coordinates state collection and storage failures across ranks.
    # The state includes frozen model parameters and rank-specific training RNG.
    try:
        save_training_checkpoint(
            checkpoint_dir=str(write_dir),
            model=model,
            optimizer_bundle=optimizer_bundle,
            progress=progress,
        )
    except CheckpointException as exc:
        # DCP has already collected rank failures. Its exception inherits
        # BaseException; expose a normal task failure to the retained trainer.
        raise RuntimeError(f"Draft phase checkpoint failed: {exc}") from exc

    def commit():
        write_checkpoint_metadata(str(write_dir), progress=progress)
        files = {}
        for file in sorted(write_dir.rglob("*")):
            if file.is_file():
                with file.open("rb") as stream:
                    os.fsync(stream.fileno())
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                files[str(file.relative_to(write_dir))] = {
                    "size": file.stat().st_size,
                    "sha256": digest,
                }
        _verify_distributed_files(write_dir, progress.saved_world_size)
        atomic_write_json(
            write_dir / _COMMIT_FILE,
            {
                "version": 1,
                "next_micro_step": progress.next_micro_step,
                "global_step": progress.global_step,
                "feature_index": feature_index.identity,
                "files": files,
            },
        )
        for directory in sorted(write_dir.rglob("*"), reverse=True):
            if directory.is_dir():
                _fsync_directory(directory)
        _fsync_directory(write_dir)
        os.rename(write_dir, destination)
        _fsync_directory(root)
        temporary_link = root / f".step_latest-{os.getpid()}"
        os.symlink(destination.name, temporary_link)
        os.replace(temporary_link, root / "step_latest")
        _fsync_directory(root)
        return str(destination)

    return _root_action(commit, control_group)
