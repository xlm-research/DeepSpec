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
    if parallel_dims.ep != 1:
        raise ValueError("DSpark expert parallelism is not supported by this recipe")
    if parallel_dims.pp_enabled and getattr(model, "pipeline_stage", None) is None:
        raise ValueError("Pipeline parallelism requires a DSpark stage split")
    if parallel_dims.spmd_backend != "spmd_types":
        raise ValueError("DSpark requires the spmd_types backend")
    if compile_config.enable:
        raise ValueError("This DSpark recipe currently requires model compile disabled")
    if parallel_dims.cp_enabled:
        cp_mesh = parallel_dims.get_mesh("cp")
        model_mesh = parallel_dims.get_optional_mesh(
            ["cp", "tp"], include_singleton_axes=True
        )._flatten("dspark_context_tensor")
        model.configure_context_parallel(
            size=parallel_dims.cp,
            rank=cp_mesh.get_local_rank(),
            group=cp_mesh.get_group(),
            model_parallel_group=model_mesh.get_group(),
            model_parallel_src_rank=int(model_mesh.mesh.flatten()[0]),
        )
    dense_storage = (
        parallel_dims.tp_enabled
        or parallel_dims.dp_replicate_enabled
        or parallel_dims.cp_enabled
    )
    if dense_storage:
        from .tensor_parallel import apply_dense_parallelism

        apply_dense_parallelism(
            model,
            parallel_dims,
            sequence_parallel=parallel_dims.tp_enabled
            and parallelism.enable_sequence_parallel,
        )
    if ac_config is not None:
        if (
            not isinstance(ac_config, SelectiveAC.Config)
            or not ac_config.preserve_rng_state
        ):
            raise ValueError(
                "DSpark currently supports SelectiveAC with RNG preservation"
            )
        ac_config.build(dump_folder=dump_folder).apply(model)
    if not dense_storage:
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
    if dense_storage:
        mesh, axes = resolve_fsdp_mesh(parallel_dims)
        kwargs.update(mesh=mesh, dp_mesh_dims=axes)
    reshard = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, pp_enabled=parallel_dims.pp_enabled
    )
    for layer in model.layers:
        if layer is not None:
            fully_shard(layer, **kwargs, reshard_after_forward=reshard)
    # The root-owned frozen head participates in hidden-state backpropagation.
    fully_shard(model, **kwargs, reshard_after_forward=False)
    disable_fsdp_gradient_division(model)
    return model
