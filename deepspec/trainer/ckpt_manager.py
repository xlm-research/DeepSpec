import os
import random
import shutil
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.distributed as dist
import yaml

from deepspec.distributed.distributed_checkpoint import (
    TrainingProgress as DistributedTrainingProgress,
    full_model_state_dict,
    save_training_checkpoint as save_distributed_training_checkpoint,
    read_checkpoint_metadata,
    write_checkpoint_metadata,
)

from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"


def discover_latest_checkpoint(checkpoint_dir):
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)


def save_train_config(*, train_config, checkpoint_dir: str) -> str:
    dest_path = os.path.join(checkpoint_dir, TRAIN_CONFIG_FILE_NAME)
    if not is_global_main_process():
        return dest_path

    ensure_dir(checkpoint_dir)
    shutil.copy(train_config._origin_config_path, dest_path)
    opts = train_config._origin_opts
    if opts:
        with open(dest_path, "a", encoding="utf-8") as handle:
            handle.write("\n\n# --opts overrides applied at save time\n")
            for opt in opts:
                handle.write(_render_opt_assignment(opt) + "\n")
    return dest_path


def _render_opt_assignment(opt: str) -> str:
    key, raw_value = opt.split("=", 1)
    head, *rest = key.split(".")
    accessors = "".join(f"[{part!r}]" for part in rest)
    value = yaml.safe_load(raw_value)
    return f"{head}{accessors} = {value!r}"


@dataclass(frozen=True)
class TrainingResumeState:
    # next_micro_step is the single source of truth for training progress;
    # global_step and current_epoch are derived from it together with
    # gradient_accumulation_steps / micro_batches_per_epoch.
    next_micro_step: int


def load_resume_draft_model(
    *,
    resume_checkpoint_dir: str,
    draft_model,
    device,
    precision_dtype,
    global_rank: int,
):
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)
    resumed_model = type(draft_model).from_pretrained(
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
    )
    if bool(getattr(draft_model, "checkpoint_excludes_embedding_head", False)):
        resumed_model.initialize_embeddings_and_head(
            embed_tokens=draft_model.embed_tokens,
            lm_head=draft_model.lm_head,
            freeze=True,
        )
    resumed_model = resumed_model.to(device=device, dtype=precision_dtype)
    resumed_model.set_embedding_head_trainable(False)
    return resumed_model


def load_training_state(
    *,
    resume_checkpoint_dir: str,
    optimizer,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    gradient_accumulation_steps: int,
    micro_batches_per_epoch: int,
) -> TrainingResumeState:
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)

    checkpoint = torch.load(state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(checkpoint["optimizer"])

    next_micro_step = int(checkpoint["next_micro_step"])
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps."
    )

    saved_rank = int(checkpoint["global_rank"])
    assert saved_rank == int(global_rank)
    
    saved_world_size = int(checkpoint["world_size"])
    assert saved_world_size == int(world_size)
    
    saved_local_batch_size = int(checkpoint["local_batch_size"])
    assert saved_local_batch_size == int(local_batch_size)

    torch.set_rng_state(checkpoint["torch_rng"])
    torch.cuda.set_rng_state(checkpoint["torch_cuda_rng"])
    np.random.set_state(checkpoint["numpy_rng"])
    random.setstate(checkpoint["python_rng"])

    global_step = next_micro_step // gradient_accumulation_steps
    current_epoch = next_micro_step // micro_batches_per_epoch + 1
    print_on_global_main(
        (
            "AUTO-RESUME from "
            f"{resume_checkpoint_dir}, next_micro_step={next_micro_step}, "
            "to force fresh run change exp_name or remove step_latest"
        )
    )
    print_on_local_main(
        f"Resumed from {resume_checkpoint_dir}: "
        f"next_micro_step={next_micro_step}, global_step={global_step}, "
        f"epoch={current_epoch}"
    )
    return TrainingResumeState(next_micro_step=next_micro_step)


def save_checkpoint(
    *,
    model,
    draft_model,
    optimizer,
    checkpoint_dir_root: str,
    train_config,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
    parallel_config,
    model_config: dict,
    micro_batches_per_epoch: int,
    partition_metadata: dict | None = None,
    atomic_commit: bool = False,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    write_dir = checkpoint_dir
    already_committed = [False]
    if atomic_commit:
        if partition_metadata is None:
            raise ValueError("atomic checkpoint commit requires partition metadata.")
        write_dir = checkpoint_dir + ".incomplete"
        preparation = [False, None]
        if is_global_main_process():
            try:
                if os.path.isdir(checkpoint_dir):
                    _verify_completed_checkpoint(
                        checkpoint_dir,
                        partition_metadata=partition_metadata,
                        next_micro_step=next_micro_step,
                    )
                    preparation[0] = True
                elif os.path.lexists(checkpoint_dir):
                    raise ValueError(
                        "Checkpoint destination is not a directory: "
                        f"{checkpoint_dir}"
                    )
                if not preparation[0]:
                    _prepare_incomplete_checkpoint(
                        write_dir,
                        partition_metadata=partition_metadata,
                    )
            except Exception as exc:
                preparation[1] = f"{type(exc).__name__}: {exc}"
        dist.broadcast_object_list(preparation, src=0)
        if preparation[1] is not None:
            raise RuntimeError(
                "Partition checkpoint preparation failed: "
                f"{preparation[1]}"
            )
        already_committed[0] = bool(preparation[0])
        if already_committed[0]:
            latest_error = [None]
            if is_global_main_process():
                try:
                    safe_symlink(
                        checkpoint_dir,
                        os.path.join(checkpoint_dir_root, "step_latest"),
                    )
                    _fsync_directory(checkpoint_dir_root)
                except Exception as exc:
                    latest_error[0] = f"{type(exc).__name__}: {exc}"
            dist.broadcast_object_list(latest_error, src=0)
            if latest_error[0] is not None:
                raise RuntimeError(
                    f"step_latest update failed: {latest_error[0]}"
                )
            dist.barrier()
            return checkpoint_dir
    config_error = [None]
    if is_global_main_process():
        try:
            ensure_dir(write_dir)
            save_train_config(
                train_config=train_config,
                checkpoint_dir=write_dir,
            )
        except Exception as exc:
            config_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(config_error, src=0)
    if config_error[0] is not None:
        raise RuntimeError(
            f"Checkpoint configuration export failed: {config_error[0]}"
        )
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=write_dir,
    )
    progress = DistributedTrainingProgress(
        next_micro_step=next_micro_step,
        global_step=global_step,
        epoch=(
            int(partition_metadata["epoch"])
            if partition_metadata is not None
            else next_micro_step // int(micro_batches_per_epoch)
        ),
        data_position=next_micro_step * int(local_batch_size),
        local_batch_size=local_batch_size,
        saved_world_size=world_size,
        parallel_config=parallel_config.to_dict(),
        model_config=model_config,
        partition_id=(
            int(partition_metadata["partition_id"])
            if partition_metadata is not None
            else None
        ),
        partition_start_next_micro_step=(
            int(partition_metadata["start_next_micro_step"])
            if partition_metadata is not None
            else None
        ),
        partition_end_next_micro_step=(
            int(partition_metadata["end_next_micro_step"])
            if partition_metadata is not None
            else None
        ),
        checkpointed=partition_metadata is not None,
    )
    save_distributed_training_checkpoint(
        checkpoint_dir=write_dir,
        model=model,
        optimizer_bundle=optimizer,
        progress=progress,
    )
    dist.barrier()
    commit_error = [None]
    if is_global_main_process():
        try:
            write_checkpoint_metadata(write_dir, progress=progress)
            if atomic_commit:
                _verify_completed_checkpoint(
                    write_dir,
                    partition_metadata=partition_metadata,
                    next_micro_step=next_micro_step,
                )
                os.rename(write_dir, checkpoint_dir)
                _fsync_directory(checkpoint_dir_root)
            safe_symlink(
                checkpoint_dir,
                os.path.join(checkpoint_dir_root, "step_latest"),
            )
            _fsync_directory(checkpoint_dir_root)
            print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
        except Exception as exc:
            commit_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(commit_error, src=0)
    if commit_error[0] is not None:
        raise RuntimeError(f"Checkpoint commit failed: {commit_error[0]}")
    dist.barrier()
    return checkpoint_dir


def _partition_transaction_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, "partition_checkpoint_transaction.json")


def _prepare_incomplete_checkpoint(
    checkpoint_dir: str,
    *,
    partition_metadata: dict,
) -> None:
    transaction = {
        "version": 1,
        "partition_id": int(partition_metadata["partition_id"]),
        "epoch": int(partition_metadata["epoch"]),
        "start_next_micro_step": int(partition_metadata["start_next_micro_step"]),
        "end_next_micro_step": int(partition_metadata["end_next_micro_step"]),
    }
    if os.path.lexists(checkpoint_dir):
        if os.path.islink(checkpoint_dir) or not os.path.isdir(checkpoint_dir):
            raise ValueError(f"Invalid incomplete checkpoint: {checkpoint_dir}")
        marker_path = _partition_transaction_path(checkpoint_dir)
        try:
            with open(marker_path, "r", encoding="utf-8") as handle:
                saved = json.load(handle)
        except FileNotFoundError as exc:
            raise ValueError(
                "Refusing to remove an unidentified incomplete checkpoint: "
                f"{checkpoint_dir}"
            ) from exc
        if saved != transaction:
            raise ValueError(
                "Incomplete checkpoint belongs to another partition; preserving "
                f"{checkpoint_dir}."
            )
        shutil.rmtree(checkpoint_dir)
    ensure_dir(checkpoint_dir)
    marker_path = _partition_transaction_path(checkpoint_dir)
    with open(marker_path, "x", encoding="utf-8") as handle:
        json.dump(transaction, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(checkpoint_dir)


def _validate_partition_checkpoint_metadata(
    metadata: dict,
    *,
    partition_metadata: dict,
    next_micro_step: int,
) -> None:
    expected = {
        "epoch": int(partition_metadata["epoch"]),
        "partition_id": int(partition_metadata["partition_id"]),
        "partition_start_next_micro_step": int(
            partition_metadata["start_next_micro_step"]
        ),
        "partition_end_next_micro_step": int(
            partition_metadata["end_next_micro_step"]
        ),
        "next_micro_step": int(next_micro_step),
        "checkpointed": True,
    }
    mismatches = {
        name: (metadata.get(name), value)
        for name, value in expected.items()
        if metadata.get(name) != value
    }
    if mismatches:
        raise ValueError(f"Partition checkpoint identity mismatch: {mismatches}")


def _verify_completed_checkpoint(
    checkpoint_dir: str,
    *,
    partition_metadata: dict,
    next_micro_step: int,
) -> None:
    metadata = read_checkpoint_metadata(checkpoint_dir)
    _validate_partition_checkpoint_metadata(
        metadata,
        partition_metadata=partition_metadata,
        next_micro_step=next_micro_step,
    )
    required = (
        TRAIN_CONFIG_FILE_NAME,
        "config.json",
        os.path.join("distributed_checkpoint", ".metadata"),
    )
    missing = [
        name
        for name in required
        if not os.path.isfile(os.path.join(checkpoint_dir, name))
        or os.path.getsize(os.path.join(checkpoint_dir, name)) <= 0
    ]
    model_files = [
        name
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".safetensors")
        and os.path.isfile(os.path.join(checkpoint_dir, name))
        and os.path.getsize(os.path.join(checkpoint_dir, name)) > 0
    ]
    if not model_files:
        missing.append("*.safetensors")
    distributed_dir = os.path.join(checkpoint_dir, "distributed_checkpoint")
    distributed_files = [
        name
        for name in os.listdir(distributed_dir)
        if name.endswith(".distcp")
        and os.path.isfile(os.path.join(distributed_dir, name))
        and os.path.getsize(os.path.join(distributed_dir, name)) > 0
    ]
    if not distributed_files:
        missing.append("distributed_checkpoint/*.distcp")
    if missing:
        raise RuntimeError(
            f"Partition checkpoint is incomplete at {checkpoint_dir}: missing={missing}."
        )


def validate_partition_checkpoint(
    checkpoint_dir: str,
    *,
    partition_metadata: dict,
    next_micro_step: int,
) -> dict:
    _verify_completed_checkpoint(
        checkpoint_dir,
        partition_metadata=partition_metadata,
        next_micro_step=next_micro_step,
    )
    return read_checkpoint_metadata(checkpoint_dir)


def _fsync_directory(path: str) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rank_training_state_path(checkpoint_dir: str, global_rank: int) -> str:
    return os.path.join(
        checkpoint_dir,
        f"training_state.rank{int(global_rank)}.pt",
    )


def _serialize_training_state(
    *,
    optimizer,
    next_micro_step: int,
    gradient_accumulation_steps: int,
    global_rank: int,
    world_size: int,
    local_batch_size: int,
):
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    return {
        "next_micro_step": int(next_micro_step),
        "optimizer": optimizer.state_dict(),
        "global_rank": int(global_rank),
        "world_size": int(world_size),
        "local_batch_size": int(local_batch_size),
        "torch_rng": torch.get_rng_state(),
        "torch_cuda_rng": torch.cuda.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }


def _full_model_state_dict(model):
    return full_model_state_dict(model)


def _save_model_checkpoint(*, model, draft_model, checkpoint_dir: str):
    state_dict = _full_model_state_dict(model)
    save_error = [None]
    if is_global_main_process():
        try:
            draft_state_dict = {}
            for key, value in state_dict.items():
                normalized_key = key
                if normalized_key.startswith("_orig_mod."):
                    normalized_key = normalized_key[len("_orig_mod.") :]
                draft_state_dict[normalized_key] = value
            filter_state_dict = getattr(
                draft_model,
                "filter_checkpoint_state_dict",
                None,
            )
            if filter_state_dict is not None:
                draft_state_dict = filter_state_dict(draft_state_dict)
            assert draft_state_dict, (
                "Failed to extract draft model state_dict from checkpoint."
            )
            draft_model.save_pretrained(
                checkpoint_dir,
                state_dict=draft_state_dict,
            )
            checkpoint_architecture_name = getattr(
                draft_model,
                "checkpoint_architecture_name",
                None,
            )
            if checkpoint_architecture_name is not None:
                draft_model.config.architectures = [
                    str(checkpoint_architecture_name)
                ]
                draft_model.config.save_pretrained(checkpoint_dir)
        except Exception as exc:
            save_error[0] = f"{type(exc).__name__}: {exc}"
    if dist.is_initialized():
        dist.broadcast_object_list(save_error, src=0)
    if save_error[0] is not None:
        raise RuntimeError(
            f"Hugging Face checkpoint export failed: {save_error[0]}"
        )
