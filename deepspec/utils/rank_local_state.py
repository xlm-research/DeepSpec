"""Communication-free, topology-specific FSDP state files."""

from __future__ import annotations

import os

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


RANK_LOCAL_STATE_FORMAT = "deepspec_rank_local_tensors_v2"


def rank_local_topology_metadata(parallel) -> dict[str, int | bool]:
    return {
        "context_parallel_size": int(parallel.context_parallel_size),
        "expert_parallel_size": int(parallel.expert_parallel_size),
        "tensor_parallel_size": int(parallel.tensor_parallel_size),
        "fsdp_size": int(parallel.fsdp_size),
        "data_parallel_size": int(parallel.data_parallel_size),
        "pure_expert_parallel": bool(parallel.pure_expert_parallel),
    }


def rank_local_state_path(
    directory: str, *, prefix: str, global_rank: int
) -> str:
    return os.path.join(
        directory,
        f"{prefix}.rank{int(global_rank):05d}.pt",
    )


def _require_fsdp(model) -> None:
    if not any(isinstance(module, FSDP) for module in model.modules()):
        raise TypeError("Rank-local state requires at least one FSDP module.")


def _named_local_tensors(model) -> dict[str, dict[str, torch.Tensor]]:
    _require_fsdp(model)
    return {
        "parameters": dict(model.named_parameters()),
        "buffers": dict(model.named_buffers()),
    }


def _snapshot_local_tensors(model) -> dict[str, dict[str, torch.Tensor]]:
    snapshot = {}
    for kind, tensors in _named_local_tensors(model).items():
        snapshot[kind] = {}
        for name, tensor in tensors.items():
            if tensor.is_meta:
                raise RuntimeError(
                    f"Cannot save meta {kind[:-1]} {name!r} in rank-local state."
                )
            # FSDP with use_orig_params=True exposes each original parameter's
            # currently owned shard here. Clone on CPU so torch.save never keeps
            # the full flat-parameter storage or a distributed tensor wrapper.
            snapshot[kind][name] = tensor.detach().to(device="cpu").clone()
    return snapshot


def _load_local_tensors(model, saved_state) -> None:
    if not isinstance(saved_state, dict):
        raise RuntimeError("Rank-local tensor state must be a mapping.")

    current_state = _named_local_tensors(model)
    validated = {}
    for kind, current_tensors in current_state.items():
        saved_tensors = saved_state.get(kind)
        if not isinstance(saved_tensors, dict):
            raise RuntimeError(f"Rank-local state is missing the {kind!r} mapping.")

        missing = sorted(set(current_tensors) - set(saved_tensors))
        unexpected = sorted(set(saved_tensors) - set(current_tensors))
        if missing or unexpected:
            raise RuntimeError(
                f"Rank-local {kind} do not match the model: "
                f"missing={missing}, unexpected={unexpected}."
            )

        mismatches = []
        for name, current_tensor in current_tensors.items():
            saved_tensor = saved_tensors[name]
            if not isinstance(saved_tensor, torch.Tensor):
                mismatches.append(f"{name}: saved value is not a tensor")
                continue
            if saved_tensor.shape != current_tensor.shape:
                mismatches.append(
                    f"{name}: shape {tuple(saved_tensor.shape)} != "
                    f"{tuple(current_tensor.shape)}"
                )
            if saved_tensor.dtype != current_tensor.dtype:
                mismatches.append(
                    f"{name}: dtype {saved_tensor.dtype} != {current_tensor.dtype}"
                )
            if current_tensor.is_meta:
                mismatches.append(f"{name}: current tensor is still on meta device")
        if mismatches:
            details = "; ".join(mismatches[:16])
            if len(mismatches) > 16:
                details += f"; and {len(mismatches) - 16} more"
            raise RuntimeError(f"Rank-local {kind} mismatch: {details}.")
        validated[kind] = (current_tensors, saved_tensors)

    with torch.no_grad():
        for current_tensors, saved_tensors in validated.values():
            for name, current_tensor in current_tensors.items():
                current_tensor.copy_(
                    saved_tensors[name].to(device=current_tensor.device)
                )


def save_rank_local_model_state(
    model,
    path: str,
    *,
    metadata: dict,
) -> None:
    """Save only tensors already owned by this rank; no collective occurs."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "format": RANK_LOCAL_STATE_FORMAT,
        "metadata": dict(metadata),
        "local_tensors": _snapshot_local_tensors(model),
    }
    temporary_path = f"{path}.tmp.{os.getpid()}"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_rank_local_model_state(
    model,
    path: str,
    *,
    expected_metadata: dict,
) -> None:
    """Load this rank's FSDP/EP/TP state without any tensor collective."""

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != RANK_LOCAL_STATE_FORMAT:
        raise RuntimeError(f"Unsupported rank-local state format in {path}.")
    metadata = payload.get("metadata", {})
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(
            f"Rank-local checkpoint topology mismatch in {path}: {mismatches}."
        )
    _load_local_tensors(model, payload.get("local_tensors"))


__all__ = [
    "RANK_LOCAL_STATE_FORMAT",
    "load_rank_local_model_state",
    "rank_local_state_path",
    "rank_local_topology_metadata",
    "save_rank_local_model_state",
]
