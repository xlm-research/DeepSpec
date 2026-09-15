"""Draft-specific parallelization, selected by native Trainer configuration."""

from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
    resolve_fsdp_mesh,
)
from torchtitan.distributed.spmd_types import annotate_replicated_parameters


def parallelize_draft(
    model,
    *,
    parallel_dims,
    training,
    parallelism,
    compile_config,
    ac_config,
    dump_folder,
):
    if any(
        degree != 1
        for degree in (
            parallel_dims.dp_replicate,
            parallel_dims.cp,
            parallel_dims.pp,
            parallel_dims.ep,
        )
    ):
        raise ValueError("This DSpark recipe currently supports FSDP2 with optional TP")
    if parallel_dims.spmd_backend != "spmd_types":
        raise ValueError("DSpark requires the spmd_types backend")
    if compile_config.enable:
        raise ValueError("This DSpark recipe currently requires model compile disabled")
    if parallel_dims.tp_enabled:
        if parallelism.enable_sequence_parallel:
            raise ValueError(
                "Draft TP currently requires sequence parallelism disabled"
            )
        from .tensor_parallel import apply_tensor_parallel

        apply_tensor_parallel(model, parallel_dims)
    if ac_config is not None:
        if (
            not isinstance(ac_config, SelectiveAC.Config)
            or not ac_config.preserve_rng_state
        ):
            raise ValueError(
                "DSpark currently supports SelectiveAC with RNG preservation"
            )
        ac_config.build(dump_folder=dump_folder).apply(model)
    if not parallel_dims.tp_enabled:
        annotate_replicated_parameters(model, parallel_dims)
    # Pure FSDP has one storage axis. The pinned PyTorch build accepts ordinary
    # parameters on this mesh; its n-D dp_mesh_dims API requires DTensors.
    mesh = parallel_dims.get_mesh("dp_shard")
    if mesh.mesh_dim_names != ("dp_shard",):
        raise ValueError("Pure FSDP requires the native dp_shard storage axis")
    policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        output_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
    )
    kwargs = {"mesh": mesh, "mp_policy": policy}
    if parallel_dims.tp_enabled:
        mesh, axes = resolve_fsdp_mesh(parallel_dims)
        kwargs.update(mesh=mesh, dp_mesh_dims=axes)
    reshard = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, pp_enabled=False
    )
    for layer in model.layers:
        fully_shard(layer, **kwargs, reshard_after_forward=reshard)
    # The root-owned frozen head participates in hidden-state backpropagation.
    fully_shard(model, **kwargs, reshard_after_forward=False)
    disable_fsdp_gradient_division(model)
    return model
