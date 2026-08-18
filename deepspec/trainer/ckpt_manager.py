import json
import os
import random
import shutil
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import FullStateDictConfig, StateDictType

from deepspec.utils import (
    ensure_dir,
    is_global_main_process,
    print_on_global_main,
    print_on_local_main,
    safe_symlink,
)


TRAIN_CONFIG_FILE_NAME = "train_config.py"
MODEL_PARALLEL_MANIFEST = "model_parallel_manifest.json"


def discover_latest_checkpoint(checkpoint_dir):
    latest_link = os.path.join(checkpoint_dir, "step_latest")
    if not (os.path.islink(latest_link) or os.path.isdir(latest_link)):
        return None
    return os.path.realpath(latest_link)


def is_model_parallel_checkpoint(checkpoint_dir: str) -> bool:
    return os.path.isfile(
        os.path.join(checkpoint_dir, MODEL_PARALLEL_MANIFEST)
    )


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
    precision_dtype,
    global_rank: int,
    parallel=None,
):
    state_path = _rank_training_state_path(resume_checkpoint_dir, global_rank)
    assert os.path.exists(state_path)
    if is_model_parallel_checkpoint(resume_checkpoint_dir):
        if parallel is None:
            raise ValueError("A model-parallel checkpoint requires its topology.")
        with open(
            os.path.join(resume_checkpoint_dir, MODEL_PARALLEL_MANIFEST),
            "r",
            encoding="utf-8",
        ) as handle:
            manifest = json.load(handle)
        expected = {
            "expert_parallel_size": int(parallel.expert_parallel_size),
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
        }
        for key, value in expected.items():
            if int(manifest[key]) != value:
                raise RuntimeError(
                    f"Checkpoint {key}={manifest[key]} does not match {value}."
                )
        if int(parallel.fsdp_rank) == 0:
            from safetensors.torch import load_file

            shard_name = (
                f"model.ep{int(parallel.expert_parallel_rank):03d}."
                f"tp{int(parallel.tensor_parallel_rank):03d}.safetensors"
            )
            state_dict = load_file(
                os.path.join(resume_checkpoint_dir, shard_name),
                device="cpu",
            )
            missing, unexpected = draft_model.load_state_dict(
                state_dict, strict=False
            )
            if missing or unexpected:
                raise RuntimeError(
                    "Model-parallel checkpoint does not match the draft model: "
                    f"missing={missing}, unexpected={unexpected}."
                )
        draft_model.set_embedding_head_trainable(False)
        return draft_model

    resumed_model = type(draft_model).from_pretrained(
        resume_checkpoint_dir,
        dtype=precision_dtype,
        attn_implementation=str(draft_model.config._attn_implementation),
    )
    # BaseTrainer wraps the resumed model with FSDP after this function.  Keep
    # it on CPU so no rank transiently materializes the full model on its GPU.
    resumed_model = resumed_model.to(device="cpu", dtype=precision_dtype)
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
    parallel=None,
) -> str:
    assert next_micro_step % gradient_accumulation_steps == 0, (
        "next_micro_step must be aligned with gradient_accumulation_steps at "
        f"checkpoint time: next_micro_step={next_micro_step}, "
        f"gradient_accumulation_steps={gradient_accumulation_steps}"
    )
    global_step = next_micro_step // gradient_accumulation_steps
    checkpoint_dir = os.path.join(checkpoint_dir_root, f"step_{global_step}")
    if is_global_main_process():
        ensure_dir(checkpoint_dir)
        save_train_config(train_config=train_config, checkpoint_dir=checkpoint_dir)
    dist.barrier()
    _save_model_checkpoint(
        model=model,
        draft_model=draft_model,
        checkpoint_dir=checkpoint_dir,
        parallel=parallel,
    )
    training_state = _serialize_training_state(
        optimizer=optimizer,
        next_micro_step=next_micro_step,
        gradient_accumulation_steps=gradient_accumulation_steps,
        global_rank=global_rank,
        world_size=world_size,
        local_batch_size=local_batch_size,
    )
    torch.save(
        training_state,
        _rank_training_state_path(checkpoint_dir, global_rank),
    )
    dist.barrier()
    if is_global_main_process():
        safe_symlink(
            checkpoint_dir,
            os.path.join(checkpoint_dir_root, "step_latest"),
        )
        print_on_global_main(f"Saved checkpoint to {checkpoint_dir}")
    dist.barrier()
    return checkpoint_dir


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


def _full_model_state_dict(model, *, rank0_only: bool = True):
    assert isinstance(model, FSDP), "training model must be wrapped in FSDP"
    state_dict_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=bool(rank0_only),
    )
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_dict_config,
    ):
        return model.state_dict()


def _normalized_state_dict(state_dict):
    normalized = {}
    for key, value in state_dict.items():
        normalized_key = key
        if normalized_key.startswith("_orig_mod."):
            normalized_key = normalized_key[len("_orig_mod.") :]
        normalized[normalized_key] = value.contiguous()
    return normalized


def _save_model_parallel_checkpoint(
    *, model, draft_model, checkpoint_dir: str, parallel
):
    """Save one full-FSDP state shard for every orthogonal ``(EP, TP)`` rank."""

    from safetensors.torch import save_file

    # Every FSDP shard group fixes (DP, CP, EP, TP), so only the DP=CP=0
    # replica needs to participate in a checkpoint unshard. Other replicas
    # hold the same trained parameters and can go directly to the global
    # barrier without materializing another full CPU state dict.
    is_checkpoint_replica = (
        int(parallel.data_parallel_rank) == 0
        and int(parallel.context_parallel_rank) == 0
    )
    state_dict = (
        _full_model_state_dict(model, rank0_only=True)
        if is_checkpoint_replica
        else {}
    )
    # ``rank0_only`` is relative to each independent FSDP shard group, so one
    # writer is produced for every orthogonal (EP, TP) partition.
    is_shard_writer = (
        is_checkpoint_replica and int(parallel.fsdp_rank) == 0
    )
    if is_shard_writer:
        shard_name = (
            f"model.ep{int(parallel.expert_parallel_rank):03d}."
            f"tp{int(parallel.tensor_parallel_rank):03d}.safetensors"
        )
        save_file(
            _normalized_state_dict(state_dict),
            os.path.join(checkpoint_dir, shard_name),
        )
    del state_dict
    dist.barrier()
    if is_global_main_process():
        draft_model.config.save_pretrained(checkpoint_dir)
        manifest = {
            "format": "deepspec_ep_tp_v1",
            "expert_parallel_size": int(parallel.expert_parallel_size),
            "tensor_parallel_size": int(parallel.tensor_parallel_size),
            "shards": [
                {
                    "expert_parallel_rank": ep_rank,
                    "tensor_parallel_rank": tp_rank,
                    "file_name": (
                        f"model.ep{ep_rank:03d}.tp{tp_rank:03d}.safetensors"
                    ),
                }
                for ep_rank in range(int(parallel.expert_parallel_size))
                for tp_rank in range(int(parallel.tensor_parallel_size))
            ],
        }
        with open(
            os.path.join(checkpoint_dir, MODEL_PARALLEL_MANIFEST),
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(manifest, handle, indent=2)
    dist.barrier()


def _save_model_checkpoint(
    *, model, draft_model, checkpoint_dir: str, parallel=None
):
    if parallel is not None and (
        int(parallel.expert_parallel_size) > 1
        or int(parallel.tensor_parallel_size) > 1
    ):
        _save_model_parallel_checkpoint(
            model=model,
            draft_model=draft_model,
            checkpoint_dir=checkpoint_dir,
            parallel=parallel,
        )
        return

    state_dict = _full_model_state_dict(model)
    if is_global_main_process():
        draft_state_dict = _normalized_state_dict(state_dict)
        assert draft_state_dict, "Failed to extract draft model state_dict from checkpoint."
        draft_model.save_pretrained(checkpoint_dir, state_dict=draft_state_dict)
