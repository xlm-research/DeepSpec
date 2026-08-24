"""Online frozen-target execution for DeepSeek-V4 DSpark training.

The target and draft models share one CP/EP/TP/FSDP topology.  Every raw
training micro-batch is evaluated exactly once by the frozen target, and the
resulting local CP hidden-state shard is consumed immediately by the draft
model.  Nothing is gathered to a leader or written to a target-cache file.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.fsdp import FSDPModule
from transformers import AutoConfig, AutoModel, FineGrainedFP8Config
from transformers.distributed.configuration_utils import DistributedConfig

from deepspec.modeling.deepseek_v4_parallel import (
    parallelize_deepseek_v4_model,
)
from deepspec.modeling.pure_ep import (
    get_pure_expert_modules,
    materialize_modules_locally,
)
from deepspec.utils import compute_context_parallel_range

from .common import TargetForwardResult
from .deepseek_v4_cp import install_target_context_parallel


def _get_target_backbone(target_model):
    candidate = getattr(target_model, "model", target_model)
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if not hasattr(candidate, "layers"):
        raise TypeError("DeepSeek-V4 target model does not expose decoder layers.")
    return candidate


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def _run_target_forward_with_hooks(
    *,
    target_model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    target_layer_ids,
):
    backbone = _get_target_backbone(target_model)
    requested = [int(layer_id) for layer_id in target_layer_ids]
    captured = {}
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            hidden = _get_hook_tensor(output)
            if hidden.ndim == 4:
                hidden = backbone.hc_head(hidden)
            captured[layer_id] = hidden.detach()

        return hook

    try:
        if -1 in requested:
            handles.append(
                backbone.embed_tokens.register_forward_hook(capture_layer(-1))
            )
        for layer_id in requested:
            if layer_id < 0:
                continue
            handles.append(
                backbone.layers[layer_id].register_forward_hook(
                    capture_layer(layer_id)
                )
            )
        output = target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=False,
            use_cache=False,
        )
        missing = [layer_id for layer_id in requested if layer_id not in captured]
        if missing:
            raise RuntimeError(f"DeepSeek-V4 target layers were not captured: {missing}.")
        hidden = torch.cat([captured[layer_id] for layer_id in requested], dim=-1)
        last_hidden = output.last_hidden_state.detach()
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()

    return TargetForwardResult(
        target_hidden_states=hidden,
        target_last_hidden_states=last_hidden,
        context_start=0,
    )


def _dequantizing_config(target_config):
    """Load an FP8 source checkpoint as ordinary frozen BF16 parameters."""

    quantization_config = getattr(target_config, "quantization_config", None)
    if quantization_config is None:
        return None
    if hasattr(quantization_config, "to_dict"):
        quantization_config = quantization_config.to_dict()
    else:
        quantization_config = dict(quantization_config)
    quant_method = str(quantization_config.get("quant_method", "")).lower()
    if quant_method not in ("fp8", "quantizationmethod.fp8"):
        return None
    quantization_config["dequantize"] = True
    return FineGrainedFP8Config(**quantization_config)


def _normalize_target_parameter_dtype(target_model) -> None:
    for parameter in target_model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)


def _build_rank_local_ep_target_model(
    *, model_name_or_path: str, target_config, topology
):
    """Load only the routed experts owned by this EP rank.

    Transformers' DeepSeek-V4 conversion pipeline dequantizes each selected
    expert before fusing w1/w3 into gate_up_proj. GroupedGemmParallel receives
    the source expert index and returns None for experts outside this rank, so
    their safetensors payloads are never materialized.
    """

    load_kwargs = {
        "config": target_config,
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
    }
    quantization_config = _dequantizing_config(target_config)
    if quantization_config is not None:
        load_kwargs["quantization_config"] = quantization_config

    ep_size = int(topology.expert_parallel_size)
    if ep_size > 1:
        if topology.expert_parallel_group is None:
            raise RuntimeError("Rank-local EP loading requires an EP process group.")
        # Only shard routed-expert parameters during checkpoint loading. The
        # framework's existing pure-EP forward owns token dispatch and routing;
        # dense modules remain replicated until the layerwise FSDP wrap below.
        target_config.base_model_ep_plan = {
            "layers.*.mlp.experts.gate_up_proj": "grouped_gemm",
            "layers.*.mlp.experts.down_proj": "grouped_gemm",
        }
        ep_mesh = DeviceMesh.from_group(
            topology.expert_parallel_group,
            "cuda",
            mesh_dim_names=("tp",),
        )
        load_kwargs.update(
            tp_plan="auto",
            device_mesh=ep_mesh,
            distributed_config=DistributedConfig(enable_expert_parallel=True),
        )

    model = AutoModel.from_pretrained(model_name_or_path, **load_kwargs)
    _normalize_target_parameter_dtype(model)

    if ep_size > 1:
        expected_local_experts = int(target_config.n_routed_experts) // ep_size
        backbone = _get_target_backbone(model)
        for layer_idx, decoder_layer in enumerate(backbone.layers):
            experts = decoder_layer.mlp.experts
            local_experts = int(experts.gate_up_proj.shape[0])
            if local_experts != expected_local_experts:
                raise RuntimeError(
                    "Rank-local target loader produced an invalid expert shard: "
                    f"layer={layer_idx}, local={local_experts}, "
                    f"expected={expected_local_experts}."
                )
            experts.num_experts = local_experts
            experts._deepspec_expert_parameters_distributed = True
    return model


def _materialize_tensor_on_device(
    *, module: nn.Module, tensor_name: str, device, is_parameter: bool
) -> torch.Tensor:
    registry = module._parameters if is_parameter else module._buffers
    tensor = registry[tensor_name]
    if tensor is None:
        raise RuntimeError(f"Cannot materialize missing tensor {tensor_name!r}.")
    materialized = (
        torch.empty_like(tensor, device=device)
        if tensor.is_meta
        else tensor.to(device=device)
    )
    if is_parameter:
        materialized = nn.Parameter(
            materialized,
            requires_grad=bool(tensor.requires_grad),
        )
    registry[tensor_name] = materialized
    return materialized


def _materialize_replicated_target_state_locally(
    *, target_model, decoder_layers, device
) -> None:
    decoder_parameter_ids = {
        id(parameter)
        for decoder_layer in decoder_layers
        for parameter in decoder_layer.parameters()
    }
    decoder_buffer_ids = {
        id(buffer)
        for decoder_layer in decoder_layers
        for buffer in decoder_layer.buffers()
    }
    for qualified_name, parameter in list(target_model.named_parameters()):
        if id(parameter) in decoder_parameter_ids:
            continue
        module_name, _, tensor_name = qualified_name.rpartition(".")
        module = target_model.get_submodule(module_name) if module_name else target_model
        materialized = _materialize_tensor_on_device(
            module=module,
            tensor_name=tensor_name,
            device=device,
            is_parameter=True,
        )
    for qualified_name, buffer in list(target_model.named_buffers()):
        if id(buffer) in decoder_buffer_ids:
            continue
        module_name, _, tensor_name = qualified_name.rpartition(".")
        module = target_model.get_submodule(module_name) if module_name else target_model
        materialized = _materialize_tensor_on_device(
            module=module,
            tensor_name=tensor_name,
            device=device,
            is_parameter=False,
        )


def _materialize_meta_module(module: nn.Module, *, device) -> None:
    if any(
        tensor is not None and tensor.is_meta
        for tensor in (
            *module.parameters(recurse=False),
            *module.buffers(recurse=False),
        )
    ):
        module.to_empty(device=device, recurse=False)


def _wrap_target_model_with_fsdp(
    *, target_model, device, topology
):
    """FSDP2-shard dense target state while keeping routed experts pure-EP."""

    backbone = _get_target_backbone(target_model)
    decoder_layers = list(backbone.layers)
    if not decoder_layers:
        raise ValueError("Target model does not expose decoder layers for FSDP.")
    _materialize_replicated_target_state_locally(
        target_model=target_model,
        decoder_layers=decoder_layers,
        device=device,
    )
    pure_expert_modules = get_pure_expert_modules(target_model)
    materialize_modules_locally(pure_expert_modules, device=device)
    ignored_params = {
        parameter
        for expert_root in pure_expert_modules
        for parameter in expert_root.parameters()
    }
    mp_policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=torch.bfloat16,
    )
    mesh = (
        topology.dense_mesh["dp_shard_cp"]
        if topology.config.dp_replicate == 1
        else topology.fsdp_mesh
    )
    for layer_idx, decoder_layer in enumerate(decoder_layers):
        fully_shard(
            decoder_layer,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=True,
            ignored_params=ignored_params,
        )
    fully_shard(
        target_model,
        mesh=mesh,
        mp_policy=mp_policy,
        reshard_after_forward=False,
        ignored_params=ignored_params,
    )
    return target_model


def _run_target_forward_context_parallel(
    *, target_model, model_inputs, target_layer_ids, topology, device
):
    cp_forward = getattr(target_model, "forward_context_parallel", None)
    if cp_forward is None:
        raise NotImplementedError(
            "The online target does not expose forward_context_parallel()."
        )
    # This model-specific method intentionally bypasses ``target_model(...)``.
    # Explicitly run the root FSDP2 materialization that its __call__ hook would
    # otherwise perform; child decoder layers still execute through __call__.
    if isinstance(target_model, FSDPModule):
        target_model.unshard()
    try:
        result = cp_forward(
            model_inputs=model_inputs,
            target_layer_ids=target_layer_ids,
            context_parallel_group=topology.context_parallel_group,
            context_parallel_rank=topology.context_parallel_rank,
            context_parallel_size=topology.context_parallel_size,
            tensor_parallel_group=topology.tensor_parallel_group,
            tensor_parallel_rank=topology.tensor_parallel_rank,
            tensor_parallel_size=topology.tensor_parallel_size,
            device=device,
        )
    finally:
        if isinstance(target_model, FSDPModule):
            target_model.reshard()
    if not isinstance(result, TargetForwardResult):
        raise TypeError(
            "forward_context_parallel() must return TargetForwardResult, got "
            f"{type(result)!r}."
        )
    sequence_length = int(model_inputs["attention_mask"].sum(dim=1)[0].item())
    expected_start, expected_end = compute_context_parallel_range(
        sequence_length=sequence_length,
        context_parallel_rank=topology.context_parallel_rank,
        context_parallel_size=topology.context_parallel_size,
    )
    local_length = int(result.target_hidden_states.shape[1])
    if (
        int(result.context_start) != expected_start
        or local_length != expected_end - expected_start
    ):
        raise RuntimeError(
            "Online target CP returned the wrong contiguous shard: expected "
            f"[{expected_start}, {expected_end}), got start={result.context_start}, "
            f"length={local_length}."
        )
    if int(result.target_last_hidden_states.shape[1]) != local_length:
        raise RuntimeError("Online target hidden-state tensors have different lengths.")
    return result


class DeepseekV4OnlineTarget:
    """Frozen DeepSeek-V4 teacher evaluated immediately before each draft step."""

    def __init__(
        self,
        *,
        model_name_or_path: str,
        target_layer_ids,
        topology,
        device,
        rank_local_cache_dir: str,
    ):
        self.model_name_or_path = str(model_name_or_path)
        self.target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
        self.topology = topology
        self.device = device
        self.rank_local_cache_dir = str(rank_local_cache_dir)
        target_config = AutoConfig.from_pretrained(self.model_name_or_path)
        if str(target_config.model_type) != "deepseek_v4":
            raise ValueError(
                "DeepseekV4OnlineTarget requires a deepseek_v4 checkpoint, got "
                f"{target_config.model_type!r}."
            )
        decoder_layer_ids = [
            layer_id for layer_id in self.target_layer_ids if layer_id >= 0
        ]
        if not decoder_layer_ids:
            raise ValueError("Online target requires at least one decoder layer.")
        retained_layers = max(decoder_layer_ids) + 1
        original_layers = int(target_config.num_hidden_layers)
        if retained_layers > original_layers:
            raise ValueError(
                f"Requested target layer {retained_layers - 1}, but the checkpoint "
                f"contains only {original_layers} decoder layers."
            )
        target_config.num_hidden_layers = retained_layers
        self.target_num_hidden_layers = retained_layers

        model = _build_rank_local_ep_target_model(
            model_name_or_path=self.model_name_or_path,
            target_config=target_config,
            topology=topology,
        )
        model = model.eval()
        model.requires_grad_(False)
        if topology.expert_parallel_size > 1 or topology.tensor_parallel_size > 1:
            model = parallelize_deepseek_v4_model(
                model,
                topology=topology,
                draft=False,
            )
        if topology.context_parallel_size > 1:
            model = install_target_context_parallel(model)
        self.model = _wrap_target_model_with_fsdp(
            target_model=model,
            device=device,
            topology=topology,
        ).eval()

    def forward_training_batch(self, batch) -> dict[str, torch.Tensor]:
        """Run one target forward and return a cache-shaped in-memory batch."""

        if int(batch["input_ids"].shape[0]) != 1:
            raise ValueError("Online DeepSeek-V4 training requires local_batch_size=1.")
        attention_mask = batch["attention_mask"]
        sequence_length = int(attention_mask.sum(dim=1)[0].item())
        if sequence_length < 1:
            raise ValueError("Online target received an empty token sequence.")
        if not bool(attention_mask[0, :sequence_length].all()) or bool(
            attention_mask[0, sequence_length:].any()
        ):
            raise ValueError("Online target requires a right-padded attention mask.")
        model_inputs = {
            "input_ids": batch["input_ids"][:, :sequence_length],
            "attention_mask": attention_mask[:, :sequence_length],
        }
        with torch.no_grad():
            if self.topology.context_parallel_size > 1:
                result = _run_target_forward_context_parallel(
                    target_model=self.model,
                    model_inputs=model_inputs,
                    target_layer_ids=self.target_layer_ids,
                    topology=self.topology,
                    device=self.device,
                )
            else:
                result = _run_target_forward_with_hooks(
                    target_model=self.model,
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    target_layer_ids=self.target_layer_ids,
                )

        context_start = int(result.context_start)
        context_len = int(result.target_hidden_states.shape[1])

        def metadata(value):
            return torch.tensor(
                [int(value)],
                dtype=torch.long,
                device=batch["input_ids"].device,
            )

        return {
            "input_ids": model_inputs["input_ids"],
            "loss_mask": batch["loss_mask"][:, :sequence_length],
            "target_hidden_states": result.target_hidden_states,
            "target_last_hidden_states": result.target_last_hidden_states,
            "context_start": metadata(context_start),
            "context_len": metadata(context_len),
            "seq_len": metadata(sequence_length),
        }

    def close(self) -> None:
        self.model = None
        torch.cuda.empty_cache()


__all__ = ["DeepseekV4OnlineTarget"]
