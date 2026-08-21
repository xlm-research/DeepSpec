# ruff: noqa: E402

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.distributed.tensor.experimental import context_parallel as torch_context_parallel
from torch.distributed.tensor.experimental._context_parallel import set_rotate_method
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, DynamicCache

from deepspec.data import ConversationCollator, MultimodalConversationCollator
from deepspec.data.parser import parse_media_uri_map_entries
from deepspec.modeling.target_adapter import (
    get_decoder_layers,
    get_final_norm,
    get_target_hidden_size,
    is_multimodal_config,
    load_target_cache_model,
)
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_source_jsonl_fingerprints,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_indices,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_diff,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)
from deepspec.utils.parallel import build_parallel_topology

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 2.10 Inductor still reads the legacy allow_tf32 flag while compiling.
torch.set_float32_matmul_precision("high")


@dataclass(frozen=True)
class TargetForwardResult:
    target_hidden_states: torch.Tensor
    target_last_hidden_states: torch.Tensor


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        first = output[0]
        if isinstance(first, torch.Tensor):
            return first
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def run_target_forward_with_hooks(
    *,
    target_model,
    model_inputs,
    target_layer_ids,
    use_cache: bool = False,
):
    layer_modules = get_decoder_layers(target_model)
    final_norm = get_final_norm(target_model)
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured_hidden_states = {}
    captured_last_hidden_state = []
    handles = []

    def capture_layer(layer_id: int):
        def hook(_module, _inputs, output):
            captured_hidden_states[layer_id] = _get_hook_tensor(output).detach()

        return hook

    def capture_decoder_input(_module, inputs):
        if not inputs:
            raise ValueError("Decoder layer pre-hook did not receive hidden states.")
        captured_hidden_states[-1] = _get_hook_tensor(inputs).detach()

    def capture_last_hidden_state(_module, _inputs, output):
        captured_last_hidden_state.append(_get_hook_tensor(output).detach())

    try:
        if -1 in target_layer_ids:
            handles.append(
                layer_modules[0].register_forward_pre_hook(capture_decoder_input)
            )
        for layer_id in target_layer_ids:
            if layer_id < 0:
                continue
            if layer_id >= len(layer_modules):
                raise ValueError(
                    f"target_layer_id {layer_id} is out of range for "
                    f"{len(layer_modules)} decoder layers."
                )
            handles.append(
                layer_modules[layer_id].register_forward_hook(capture_layer(layer_id))
            )
        handles.append(final_norm.register_forward_hook(capture_last_hidden_state))

        with torch.no_grad():
            target_output = target_model(
                **model_inputs,
                output_hidden_states=False,
                use_cache=use_cache,
                return_dict=True,
            )
            if captured_last_hidden_state:
                target_last_hidden_states = captured_last_hidden_state[-1]
            else:
                target_last_hidden_states = target_output.last_hidden_state.detach()
            missing = [
                layer_id
                for layer_id in target_layer_ids
                if layer_id not in captured_hidden_states
            ]
            if missing:
                raise RuntimeError(f"Failed to capture target layers: {missing}")
            target_hidden_states = torch.cat(
                [captured_hidden_states[layer_id] for layer_id in target_layer_ids],
                dim=-1,
            )
            sequence_tensor = model_inputs.get("input_ids")
            if sequence_tensor is None:
                sequence_tensor = model_inputs["inputs_embeds"]
            expected_sequence_length = int(sequence_tensor.shape[1])
            if (
                int(target_hidden_states.shape[1]) != expected_sequence_length
                or int(target_last_hidden_states.shape[1]) != expected_sequence_length
            ):
                raise RuntimeError(
                    "Target decoder hidden-state length must match processor "
                    f"input_ids length ({expected_sequence_length}), got "
                    f"{target_hidden_states.shape[1]} and "
                    f"{target_last_hidden_states.shape[1]}. This target family "
                    "needs a custom TargetModelAdapter before it can use the "
                    "shared multimodal cache pipeline."
                )
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()
        captured_last_hidden_state.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
    )


def _wrap_target_model_with_fsdp(
    *, target_model, device, process_group, layerwise_only: bool = False
):
    """Shard target parameters layer-by-layer for forward-only cache generation."""
    decoder_layers = list(get_decoder_layers(target_model))
    if not decoder_layers:
        raise ValueError("Target model does not expose any decoder layers for FSDP.")
    common_kwargs = dict(
        process_group=process_group,
        device_id=device,
        limit_all_gathers=True,
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        use_orig_params=True,
    )
    if layerwise_only:
        # Keep embeddings and the vision tower directly callable so CP can
        # build full multimodal embeddings/MRoPE positions before slicing.
        layer_container = get_decoder_layers(target_model)
        for layer_idx, layer in enumerate(list(layer_container)):
            layer_container[layer_idx] = FSDP(layer, **common_kwargs)
        return target_model

    decoder_layer_classes = {type(layer) for layer in decoder_layers}
    return FSDP(
        target_model,
        auto_wrap_policy=ModuleWrapPolicy(decoder_layer_classes),
        **common_kwargs,
    )


def _build_dummy_model_inputs(*, device):
    """Keep FSDP collectives aligned when another rank has no valid record."""
    return {
        "input_ids": torch.zeros((1, 1), dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, 1), dtype=torch.long, device=device),
    }


def _prepare_cp_embeddings_and_positions(*, target_model, model_inputs):
    """Reproduce the Qwen3.6 outer-model preprocessing before CP slicing."""
    input_ids = model_inputs["input_ids"]
    inputs_embeds = target_model.get_input_embeddings()(input_ids)

    pixel_values = model_inputs.get("pixel_values")
    if pixel_values is not None:
        image_outputs = target_model.get_image_features(
            pixel_values,
            model_inputs.get("image_grid_thw"),
            return_dict=True,
        )
        image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        image_mask, _ = target_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    pixel_values_videos = model_inputs.get("pixel_values_videos")
    if pixel_values_videos is not None:
        video_outputs = target_model.get_video_features(
            pixel_values_videos,
            model_inputs.get("video_grid_thw"),
            return_dict=True,
        )
        video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
            inputs_embeds.device, inputs_embeds.dtype
        )
        _, video_mask = target_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    compute_3d_position_ids = getattr(target_model, "compute_3d_position_ids", None)
    if compute_3d_position_ids is not None:
        position_ids = compute_3d_position_ids(
            input_ids=input_ids,
            image_grid_thw=model_inputs.get("image_grid_thw"),
            video_grid_thw=model_inputs.get("video_grid_thw"),
            inputs_embeds=inputs_embeds,
            attention_mask=model_inputs.get("attention_mask"),
            past_key_values=None,
            mm_token_type_ids=model_inputs.get("mm_token_type_ids"),
        )
    else:
        position_ids = None
    # Qwen3.6 intentionally returns None for text-only records and lets its
    # text backbone infer flat positions. CP slicing needs an explicit global
    # tensor, so construct the equivalent flat positions here.
    if position_ids is None:
        position_ids = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
    return inputs_embeds, position_ids


_CACHE_DTYPES = {
    torch.bfloat16: 0,
    torch.float32: 1,
    torch.float16: 2,
}
_CACHE_DTYPES_BY_CODE = {value: key for key, value in _CACHE_DTYPES.items()}


class _FullAttentionCPUOffloadDynamicCache(DynamicCache):
    """Keep growing full-attention KV tensors on CPU between layer calls.

    Transformers' generic cache offloader assumes adjacent attention layers:
    one layer prefetches the next. Qwen3.6 has three DeltaNet layers between
    full-attention layers, so synchronously stage the current KV layer instead.
    The GPU tensors returned from ``update`` remain alive for the current SDPA
    call, while the cache-owned copy is moved back to CPU for later forwards.
    """

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        layer = self.layers[layer_idx]
        if getattr(layer, "is_initialized", False):
            if layer.keys.device != key_states.device:
                layer.keys = layer.keys.to(key_states.device)
                layer.values = layer.values.to(value_states.device)
                # Cache layers remember their execution device for later moves.
                layer.device = key_states.device

        keys, values = super().update(
            key_states, value_states, layer_idx, *args, **kwargs
        )
        layer.keys = keys.detach().to("cpu", non_blocking=False)
        layer.values = values.detach().to("cpu", non_blocking=False)
        return keys, values


def _build_target_cache(*, config, cpu_offload: bool):
    cache_cls = (
        _FullAttentionCPUOffloadDynamicCache if cpu_offload else DynamicCache
    )
    return cache_cls(config=config)


class _NativeContextParallelCache(DynamicCache):
    """Keep only Qwen3.6 linear-attention state during native CP prefill.

    PyTorch Context Parallel rotates each full-attention layer's local K/V
    shards while that layer is running. Retaining those same K/V tensors in a
    ``DynamicCache`` until the end of the model would make memory grow once per
    full-attention layer and defeat CP. Linear-attention conv/recurrent states
    are still cached because they are the compact prefix summary handed across
    the head/tail CP schedule.
    """

    def __init__(self, *, config):
        super().__init__(config=config)
        decoder_config = config.get_text_config(decoder=True)
        self._full_attention_layers = {
            layer_idx
            for layer_idx, layer_type in enumerate(decoder_config.layer_types)
            if layer_type == "full_attention"
        }

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        if int(layer_idx) in self._full_attention_layers:
            return key_states, value_states
        return super().update(
            key_states,
            value_states,
            layer_idx,
            *args,
            **kwargs,
        )


def _pad_native_cp_inputs(
    *, inputs_embeds, position_ids, attention_mask, context_parallel_size: int
):
    """Pad to the head/tail Ring-Attention alignment without masking the tail.

    Appended tokens are causally after every real token, so they cannot alter
    real-token outputs. They intentionally remain unmasked: PyTorch's native
    causal CP path requires ``is_causal=True`` and equal head/tail shards.
    """

    sequence_length = int(inputs_embeds.shape[1])
    if attention_mask is None:
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], sequence_length),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    if int(attention_mask.shape[-1]) != sequence_length:
        raise ValueError(
            "Native target CP requires attention_mask to match inputs_embeds: "
            f"{attention_mask.shape} versus {inputs_embeds.shape}."
        )
    if not bool(torch.all(attention_mask != 0).item()):
        raise ValueError(
            "Native target CP currently requires an unpadded local_batch_size=1 "
            "record. Compact padding before target-cache preparation."
        )

    alignment = 2 * int(context_parallel_size)
    padded_length = (
        (sequence_length + alignment - 1) // alignment
    ) * alignment
    pad_tokens = padded_length - sequence_length
    if pad_tokens:
        inputs_embeds = torch.cat(
            [
                inputs_embeds,
                torch.zeros(
                    (
                        inputs_embeds.shape[0],
                        pad_tokens,
                        inputs_embeds.shape[-1],
                    ),
                    dtype=inputs_embeds.dtype,
                    device=inputs_embeds.device,
                ),
            ],
            dim=1,
        )
        position_pad_shape = list(position_ids.shape)
        position_pad_shape[-1] = pad_tokens
        position_ids = torch.cat(
            [
                position_ids,
                torch.zeros(
                    position_pad_shape,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                ),
            ],
            dim=-1,
        )
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (attention_mask.shape[0], pad_tokens),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
            ],
            dim=-1,
        )
    return (
        inputs_embeds.contiguous(),
        position_ids.contiguous(),
        attention_mask.contiguous(),
        sequence_length,
        padded_length,
    )


def _send_optional_cache_tensor(*, tensor, dst, gpu_group, cpu_group):
    present = torch.tensor([int(tensor is not None)], dtype=torch.int64)
    dist.send(present, dst=dst, group=cpu_group)
    if tensor is not None:
        if tensor.dtype not in _CACHE_DTYPES:
            raise TypeError(f"Unsupported CP cache dtype: {tensor.dtype}")
        use_cpu_transport = tensor.device.type == "cpu"
        header = torch.tensor(
            [
                tensor.ndim,
                _CACHE_DTYPES[tensor.dtype],
                int(use_cpu_transport),
            ],
            dtype=torch.int64,
        )
        shape = torch.tensor(tensor.shape, dtype=torch.int64)
        dist.send(header, dst=dst, group=cpu_group)
        dist.send(shape, dst=dst, group=cpu_group)
        dist.send(
            tensor.contiguous(),
            dst=dst,
            group=cpu_group if use_cpu_transport else gpu_group,
        )


def _recv_optional_cache_tensor(*, src, device, gpu_group, cpu_group):
    present = torch.empty(1, dtype=torch.int64)
    dist.recv(present, src=src, group=cpu_group)
    if int(present.item()) == 0:
        return None
    header = torch.empty(3, dtype=torch.int64)
    dist.recv(header, src=src, group=cpu_group)
    ndim, dtype_code, use_cpu_transport = (
        int(value) for value in header.tolist()
    )
    shape = torch.empty(ndim, dtype=torch.int64)
    dist.recv(shape, src=src, group=cpu_group)
    tensor = torch.empty(
        tuple(int(value) for value in shape.tolist()),
        dtype=_CACHE_DTYPES_BY_CODE[dtype_code],
        device="cpu" if use_cpu_transport else device,
    )
    dist.recv(
        tensor,
        src=src,
        group=cpu_group if use_cpu_transport else gpu_group,
    )
    return tensor


def _send_linear_attention_cache_layer(
    *, cache, layer_idx: int, dst, gpu_group, cpu_group
):
    layer = cache.layers[int(layer_idx)]
    if not hasattr(layer, "is_conv_states_initialized"):
        raise TypeError(
            f"Target cache layer {layer_idx} is not a linear-attention cache."
        )
    conv_states = (
        layer.conv_states if layer.is_conv_states_initialized else None
    )
    recurrent_states = (
        layer.recurrent_states
        if layer.is_recurrent_states_initialized
        else None
    )
    _send_optional_cache_tensor(
        tensor=conv_states,
        dst=dst,
        gpu_group=gpu_group,
        cpu_group=cpu_group,
    )
    _send_optional_cache_tensor(
        tensor=recurrent_states,
        dst=dst,
        gpu_group=gpu_group,
        cpu_group=cpu_group,
    )


def _recv_linear_attention_cache_layer(
    *, cache, layer_idx: int, src, device, gpu_group, cpu_group
):
    conv_states = _recv_optional_cache_tensor(
        src=src,
        device=device,
        gpu_group=gpu_group,
        cpu_group=cpu_group,
    )
    recurrent_states = _recv_optional_cache_tensor(
        src=src,
        device=device,
        gpu_group=gpu_group,
        cpu_group=cpu_group,
    )
    if conv_states is None or recurrent_states is None:
        raise RuntimeError(
            f"Missing Qwen3.6 linear-attention state for layer {layer_idx}."
        )
    layer = cache.layers[int(layer_idx)]
    if not hasattr(layer, "is_conv_states_initialized"):
        raise TypeError(
            f"Target cache layer {layer_idx} is not a linear-attention cache."
        )
    if not layer.is_conv_states_initialized:
        layer.lazy_initialization(conv_states=conv_states)
    if not layer.is_recurrent_states_initialized:
        layer.lazy_initialization(recurrent_states=recurrent_states)
    if layer.conv_states.shape != conv_states.shape:
        raise RuntimeError(
            "Received linear-attention conv state has a different shape: "
            f"{conv_states.shape} versus {layer.conv_states.shape}."
        )
    if layer.recurrent_states.shape != recurrent_states.shape:
        raise RuntimeError(
            "Received linear-attention recurrent state has a different shape: "
            f"{recurrent_states.shape} versus {layer.recurrent_states.shape}."
        )
    layer.conv_states.copy_(conv_states)
    layer.recurrent_states.copy_(recurrent_states)
    layer.has_previous_state = True


def _send_target_cache_state(*, cache, dst, gpu_group, cpu_group):
    for layer in cache.layers:
        if hasattr(layer, "conv_states") or hasattr(
            layer, "is_conv_states_initialized"
        ):
            conv_states = (
                getattr(layer, "conv_states", None)
                if getattr(layer, "is_conv_states_initialized", False)
                else None
            )
            recurrent_states = (
                getattr(layer, "recurrent_states", None)
                if getattr(layer, "is_recurrent_states_initialized", False)
                else None
            )
            _send_optional_cache_tensor(
                tensor=conv_states,
                dst=dst,
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )
            _send_optional_cache_tensor(
                tensor=recurrent_states,
                dst=dst,
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )
        else:
            keys = (
                getattr(layer, "keys", None)
                if getattr(layer, "is_initialized", False)
                else None
            )
            values = (
                getattr(layer, "values", None)
                if getattr(layer, "is_initialized", False)
                else None
            )
            _send_optional_cache_tensor(
                tensor=keys,
                dst=dst,
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )
            _send_optional_cache_tensor(
                tensor=values,
                dst=dst,
                gpu_group=gpu_group,
                cpu_group=cpu_group,
            )


def _recv_target_cache_state(
    *, cache, src, device, gpu_group, cpu_group
):
    for layer in cache.layers:
        first = _recv_optional_cache_tensor(
            src=src,
            device=device,
            gpu_group=gpu_group,
            cpu_group=cpu_group,
        )
        second = _recv_optional_cache_tensor(
            src=src,
            device=device,
            gpu_group=gpu_group,
            cpu_group=cpu_group,
        )
        if hasattr(layer, "is_conv_states_initialized"):
            if first is not None:
                layer.lazy_initialization(conv_states=first)
                layer.conv_states.copy_(first)
                layer.has_previous_state = True
            if second is not None:
                layer.lazy_initialization(recurrent_states=second)
                layer.recurrent_states.copy_(second)
        elif first is not None:
            layer.lazy_initialization(first, second)
            layer.keys = first
            layer.values = second


def _split_linear_attention_mask(attention_mask, split_size: int):
    if attention_mask is None:
        return None, None
    if attention_mask.ndim != 2 or int(attention_mask.shape[-1]) != 2 * split_size:
        raise ValueError(
            "Native Qwen3.6 CP expects a 2-D linear-attention mask with "
            f"local length {2 * split_size}, got {attention_mask.shape}."
        )
    return (
        attention_mask[:, :split_size],
        attention_mask[:, split_size:],
    )


def _run_native_cp_linear_attention(
    *,
    original_forward,
    layer_idx: int,
    hidden_states,
    cache_params,
    attention_mask,
    topology,
    device,
    forward_kwargs,
):
    """Run one FLA layer in causal head/tail order across the CP mesh.

    PyTorch's causal CP load balancer gives rank ``r`` original chunk ``r``
    followed by chunk ``2 * CP - r - 1``. Qwen3.6's recurrent DeltaNet state
    therefore scans ranks forward for the first half and backward for the
    second half. Full-attention layers remain concurrent and use native Ring
    Attention; only this compact FLA state is serialized.
    """

    local_length = int(hidden_states.shape[1])
    if local_length % 2 != 0:
        raise ValueError(
            "Native causal CP requires two equal local head/tail chunks, got "
            f"local sequence length {local_length}."
        )
    if not isinstance(cache_params, _NativeContextParallelCache):
        raise TypeError(
            "Native target CP requires _NativeContextParallelCache for FLA."
        )
    split_size = local_length // 2
    head_hidden = hidden_states[:, :split_size]
    tail_hidden = hidden_states[:, split_size:]
    head_mask, tail_mask = _split_linear_attention_mask(
        attention_mask,
        split_size,
    )

    cp_rank = topology.context_parallel_rank
    cp_size = topology.context_parallel_size
    if cp_rank > 0:
        _recv_linear_attention_cache_layer(
            cache=cache_params,
            layer_idx=layer_idx,
            src=topology.global_rank - topology.fsdp_size,
            device=device,
            gpu_group=topology.context_parallel_group,
            cpu_group=topology.context_parallel_cpu_group,
        )
    head_output = original_forward(
        hidden_states=head_hidden,
        cache_params=cache_params,
        attention_mask=head_mask,
        **forward_kwargs,
    )
    if cp_rank < cp_size - 1:
        _send_linear_attention_cache_layer(
            cache=cache_params,
            layer_idx=layer_idx,
            dst=topology.global_rank + topology.fsdp_size,
            gpu_group=topology.context_parallel_group,
            cpu_group=topology.context_parallel_cpu_group,
        )

    if cp_rank < cp_size - 1:
        _recv_linear_attention_cache_layer(
            cache=cache_params,
            layer_idx=layer_idx,
            src=topology.global_rank + topology.fsdp_size,
            device=device,
            gpu_group=topology.context_parallel_group,
            cpu_group=topology.context_parallel_cpu_group,
        )
    tail_output = original_forward(
        hidden_states=tail_hidden,
        cache_params=cache_params,
        attention_mask=tail_mask,
        **forward_kwargs,
    )
    if cp_rank > 0:
        _send_linear_attention_cache_layer(
            cache=cache_params,
            layer_idx=layer_idx,
            dst=topology.global_rank - topology.fsdp_size,
            gpu_group=topology.context_parallel_group,
            cpu_group=topology.context_parallel_cpu_group,
        )
    return torch.cat([head_output, tail_output], dim=1)


@contextmanager
def _patch_native_cp_linear_attention(*, target_model, topology, device):
    patched_modules = []
    for decoder_layer in get_decoder_layers(target_model):
        unwrapped_layer = (
            decoder_layer.module
            if isinstance(decoder_layer, FSDP)
            else decoder_layer
        )
        linear_attention = getattr(unwrapped_layer, "linear_attn", None)
        if linear_attention is None:
            continue
        original_forward = linear_attention.forward
        layer_idx = int(linear_attention.layer_idx)

        def cp_forward(
            hidden_states,
            cache_params=None,
            attention_mask=None,
            _original_forward=original_forward,
            _layer_idx=layer_idx,
            **kwargs,
        ):
            return _run_native_cp_linear_attention(
                original_forward=_original_forward,
                layer_idx=_layer_idx,
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                topology=topology,
                device=device,
                forward_kwargs=kwargs,
            )

        linear_attention.forward = cp_forward
        patched_modules.append((linear_attention, original_forward))
    try:
        yield
    finally:
        for linear_attention, original_forward in patched_modules:
            linear_attention.forward = original_forward


def _run_target_forward_native_cp(
    *,
    target_model,
    full_embeddings,
    full_position_ids,
    attention_mask,
    target_layer_ids,
    topology,
    context_parallel_mesh,
    device,
):
    (
        full_embeddings,
        full_position_ids,
        attention_mask,
        _original_sequence_length,
        _padded_sequence_length,
    ) = _pad_native_cp_inputs(
        inputs_embeds=full_embeddings,
        position_ids=full_position_ids,
        attention_mask=attention_mask,
        context_parallel_size=topology.context_parallel_size,
    )
    buffers = [full_embeddings, full_position_ids, attention_mask]
    buffer_seq_dims = [1, full_position_ids.ndim - 1, 1]
    target_cache = _NativeContextParallelCache(config=target_model.config)

    with torch_context_parallel(
        context_parallel_mesh,
        buffers=buffers,
        buffer_seq_dims=buffer_seq_dims,
        no_restore_buffers=set(buffers),
    ):
        # ``resize_`` keeps the original full-sequence CUDA storage capacity.
        # Rebind each no-restore buffer to compact local-shard storage before
        # the decoder starts so CP also releases the full embedding allocation.
        for buffer in buffers:
            buffer.set_(buffer.clone())
        with _patch_native_cp_linear_attention(
            target_model=target_model,
            topology=topology,
            device=device,
        ):
            target_result = run_target_forward_with_hooks(
                target_model=target_model,
                model_inputs={
                    "inputs_embeds": full_embeddings,
                    "position_ids": full_position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": target_cache,
                },
                target_layer_ids=target_layer_ids,
                use_cache=True,
            )

    del target_cache, full_embeddings, full_position_ids, attention_mask
    # Keep the native head/tail shard on its owning CP rank.  Stage 2 opens
    # the matching per-rank cache index and consumes this tensor directly.
    return target_result


def _run_target_forward_micro_chunked(
    *,
    target_model,
    local_embeddings,
    local_position_ids,
    attention_mask,
    global_chunk_start: int,
    target_layer_ids,
    target_cache,
    micro_chunk_size: int,
    topology,
    device,
):
    """Prefill one CP shard through bounded target workspaces.

    FSDP ranks are allowed to own different-length samples, but every rank in
    an FSDP group must enter every wrapped decoder layer the same number of
    times. The maximum micro-step count is therefore agreed by the FSDP group;
    ranks that finish early execute one-token forwards on a private dummy cache.
    """

    local_length = int(local_embeddings.shape[1])
    local_micro_steps = (
        local_length + int(micro_chunk_size) - 1
    ) // int(micro_chunk_size)
    aligned_micro_steps = torch.tensor(
        [local_micro_steps], dtype=torch.int64, device=device
    )
    if topology.fsdp_size > 1:
        dist.all_reduce(
            aligned_micro_steps,
            op=dist.ReduceOp.MAX,
            group=topology.fsdp_group,
        )
    aligned_micro_steps = int(aligned_micro_steps.item())

    hidden_pieces = []
    last_hidden_pieces = []
    dummy_cache = None
    dummy_steps = 0
    hidden_size = int(local_embeddings.shape[-1])
    for micro_idx in range(aligned_micro_steps):
        local_start = micro_idx * int(micro_chunk_size)
        local_end = min(local_start + int(micro_chunk_size), local_length)
        if local_start < local_length:
            # FSDP layer all-gathers and growing causal-attention buffers leave
            # differently sized cached blocks behind. Return unused blocks at
            # the micro-forward boundary so the next large SDPA allocation has
            # contiguous driver-visible headroom.
            torch.cuda.empty_cache()
            global_end = global_chunk_start + local_end
            micro_embeddings = local_embeddings[
                :, local_start:local_end
            ].to(device, non_blocking=False)
            micro_position_ids = local_position_ids[
                ..., local_start:local_end
            ].to(device, non_blocking=False)
            micro_inputs = {
                "inputs_embeds": micro_embeddings,
                "position_ids": micro_position_ids,
                "attention_mask": attention_mask[:, :global_end],
                "past_key_values": target_cache,
            }
            micro_result = run_target_forward_with_hooks(
                target_model=target_model,
                model_inputs=micro_inputs,
                target_layer_ids=target_layer_ids,
                use_cache=True,
            )
            hidden_pieces.append(
                micro_result.target_hidden_states.detach().to("cpu")
            )
            last_hidden_pieces.append(
                micro_result.target_last_hidden_states.detach().to("cpu")
            )
            del micro_result, micro_inputs, micro_embeddings, micro_position_ids
        else:
            if dummy_cache is None:
                dummy_cache = DynamicCache(config=target_model.config)
            dummy_steps += 1
            dummy_inputs = {
                "inputs_embeds": torch.zeros(
                    (1, 1, hidden_size),
                    dtype=local_embeddings.dtype,
                    device=device,
                ),
                "position_ids": torch.full(
                    (1, 1),
                    fill_value=dummy_steps - 1,
                    dtype=torch.long,
                    device=device,
                ),
                "attention_mask": torch.ones(
                    (1, dummy_steps), dtype=torch.long, device=device
                ),
                "past_key_values": dummy_cache,
            }
            # The output and dummy cache never enter the real sample state.
            run_target_forward_with_hooks(
                target_model=target_model,
                model_inputs=dummy_inputs,
                target_layer_ids=target_layer_ids,
                use_cache=True,
            )

    if not hidden_pieces:
        raise RuntimeError("A CP shard cannot be empty.")
    return TargetForwardResult(
        target_hidden_states=torch.cat(hidden_pieces, dim=1),
        target_last_hidden_states=torch.cat(last_hidden_pieces, dim=1),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Training JSONL path. Repeat this argument to use multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--min-loss-tokens", type=int, default=14)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    parser.add_argument("--local-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--fsdp",
        action="store_true",
        help=(
            "Use layer-wise FSDP FULL_SHARD for target-model forward passes. "
            "All ranks remain sample-parallel but execute each forward step "
            "collectively."
        ),
    )
    parser.add_argument(
        "--fsdp-size",
        type=int,
        default=None,
        help=(
            "Number of ranks in each target FSDP shard group. Defaults to "
            "world_size / context_parallel_size."
        ),
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help=(
            "Use this many ranks for PyTorch native causal Context Parallel. "
            "Full-attention K/V shards rotate with Ring Attention; Qwen3.6 "
            "DeltaNet passes only compact recurrent state."
        ),
    )
    parser.add_argument(
        "--target-micro-chunk-size",
        type=int,
        default=0,
        help=(
            "Legacy cached-prefill micro chunk size. Native Context Parallel "
            "requires 0 because Ring Attention already shards the sequence."
        ),
    )
    parser.add_argument(
        "--target-cache-cpu-offload",
        choices=("true", "false"),
        default="false",
        help=(
            "Keep growing full-attention KV cache layers on CPU between "
            "target forwards. This trades PCIe traffic for lower peak CUDA "
            "memory and is recommended for 256K target-cache generation."
        ),
    )
    parser.add_argument(
        "--media-root",
        default=None,
        help="Optional root used to resolve relative image and video paths.",
    )
    parser.add_argument(
        "--media-uri-map",
        action="append",
        default=[],
        metavar="SOURCE_PREFIX=REPLACEMENT_PREFIX",
        help=(
            "Rewrite image/video URI prefixes before loading media. Repeat for "
            "multiple mappings; the longest matching prefix wins."
        ),
    )
    cli_args = parser.parse_args()
    try:
        cli_args.media_uri_map = parse_media_uri_map_entries(cli_args.media_uri_map)
    except (TypeError, ValueError) as exc:
        parser.error(str(exc))
    config = parse_opts_to_config(cli_args.opts, load_config(cli_args.config))
    return cli_args, config


def _write_manifest(
    *,
    output_dir: str,
    num_samples: int,
    index_files,
    config,
    train_data_paths,
    target_layer_ids,
    hidden_size: int,
    min_loss_tokens: int,
    multimodal: bool,
    processor_class: str | None,
    media_root: str | None,
    media_uri_map,
    context_parallel_size: int,
    fsdp_size: int,
    target_micro_chunk_size: int,
    target_cache_cpu_offload: bool,
    stores_target_last_hidden_states: bool,
    shards,
):
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(config.model.target_model_name_or_path),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "source_jsonl_fingerprints": build_source_jsonl_fingerprints(
                train_data_paths
            ),
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(min_loss_tokens),
            "multimodal": bool(multimodal),
            "processor_class": processor_class,
            "media_root": media_root,
            "media_uri_map": media_uri_map,
            "target_context_parallel_size": int(context_parallel_size),
            "cache_context_parallel_size": int(context_parallel_size),
            "context_layout": (
                "native_head_tail" if context_parallel_size > 1 else "contiguous"
            ),
            "index_files": [str(file_name) for file_name in index_files],
            "target_context_parallel_implementation": (
                "pytorch_native" if context_parallel_size > 1 else "disabled"
            ),
            "target_fsdp_size": int(fsdp_size),
            "target_micro_chunk_size": int(target_micro_chunk_size),
            "target_cache_cpu_offload": bool(target_cache_cpu_offload),
            "stores_target_last_hidden_states": bool(
                stores_target_last_hidden_states
            ),
            "project_name": (
                str(config.get("project_name"))
                if config.get("project_name") is not None
                else None
            ),
            "exp_name": (
                str(config.get("exp_name"))
                if config.get("exp_name") is not None
                else None
            ),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def _print_prepare_progress(*, global_rank: int, processed_samples: int, total_samples: int):
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    media_root = (
        os.path.abspath(cli_args.media_root) if cli_args.media_root is not None else None
    )
    media_uri_map = cli_args.media_uri_map
    target_config = AutoConfig.from_pretrained(
        config.model.target_model_name_or_path,
    )
    configured_multimodal = config.data.get("multimodal")
    multimodal = (
        is_multimodal_config(target_config)
        if configured_multimodal is None
        else bool(configured_multimodal)
    )
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist(local_rank)
    context_parallel_size = int(cli_args.context_parallel_size)
    target_micro_chunk_size = int(cli_args.target_micro_chunk_size)
    target_cache_cpu_offload = cli_args.target_cache_cpu_offload == "true"
    stores_target_last_hidden_states = bool(
        config.data.get("store_target_last_hidden_states", True)
    )
    if target_micro_chunk_size < 0:
        raise ValueError("--target-micro-chunk-size cannot be negative.")
    if context_parallel_size > 1 and not cli_args.fsdp:
        raise ValueError("Target context parallelism requires --fsdp.")
    if context_parallel_size > 1 and target_micro_chunk_size != 0:
        raise ValueError(
            "PyTorch native target Context Parallel requires "
            "--target-micro-chunk-size 0."
        )
    if context_parallel_size > 1 and target_cache_cpu_offload:
        raise ValueError(
            "PyTorch native target Context Parallel keeps full-attention K/V "
            "sequence-sharded and requires --target-cache-cpu-offload false."
        )
    fsdp_size = (
        world_size // context_parallel_size
        if cli_args.fsdp_size is None
        else int(cli_args.fsdp_size)
    )
    topology = build_parallel_topology(
        context_parallel_size=context_parallel_size,
        fsdp_size=fsdp_size,
        create_fsdp_groups=True,
    )
    context_parallel_mesh = None
    if context_parallel_size > 1:
        context_parallel_mesh = DeviceMesh.from_group(
            topology.context_parallel_group,
            "cuda",
            mesh_dim_names=("context",),
        )
        # alltoall rotates one K/V shard at a time. The default allgather
        # materializes all shards and recreates the long-context memory peak.
        set_rotate_method("alltoall")
    output_dir = os.path.abspath(cli_args.output_dir)
    print_on_local_main(json.dumps(config, indent=4, cls=CustomJSONEncoder), flush=True)
    print_on_local_main(
        json.dumps(
            {
                "train_data_path": train_data_paths,
                "output_dir": output_dir,
                "target_layer_ids": target_layer_ids,
                "min_loss_tokens": min_loss_tokens,
                "max_shard_bytes": int(cli_args.max_shard_bytes),
                "local_batch_size": int(cli_args.local_batch_size),
                "num_workers": int(cli_args.num_workers),
                "fsdp": bool(cli_args.fsdp),
                "fsdp_size": fsdp_size,
                "context_parallel_size": context_parallel_size,
                "context_parallel_implementation": (
                    "pytorch_native" if context_parallel_size > 1 else "disabled"
                ),
                "target_micro_chunk_size": target_micro_chunk_size,
                "target_cache_cpu_offload": target_cache_cpu_offload,
                "stores_target_last_hidden_states": (
                    stores_target_last_hidden_states
                ),
                "sample_parallel_size": topology.sample_parallel_size,
                "multimodal": multimodal,
                "media_root": media_root,
                "media_uri_map": media_uri_map,
            },
            indent=4,
        ),
        flush=True,
    )
    if global_rank == 0:
        prepare_target_cache_output_dir(output_dir)
    dist.barrier()

    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)

    local_start, local_end = compute_local_sample_range(
        num_samples=len(dataset),
        rank=topology.sample_parallel_rank,
        world_size=topology.sample_parallel_size,
    )
    local_total_samples = local_end - local_start
    if cli_args.fsdp and int(cli_args.local_batch_size) != 1:
        raise ValueError("Target-cache FSDP currently requires --local-batch-size 1.")
    local_indices = list(range(local_start, local_end))
    if cli_args.fsdp:
        # Every rank must issue the same number of FSDP parameter collectives.
        max_local_steps = (
            len(dataset) + topology.sample_parallel_size - 1
        ) // topology.sample_parallel_size
        padding_steps = max_local_steps - len(local_indices)
        if padding_steps:
            padding_index = local_indices[0] if local_indices else 0
            local_indices.extend([padding_index] * padding_steps)
    else:
        padding_steps = 0

    local_subset = Subset(dataset, local_indices)
    processor = None
    if multimodal:
        processor = AutoProcessor.from_pretrained(
            config.model.target_model_name_or_path,
        )
        train_collator = MultimodalConversationCollator(
            processor=processor,
            chat_template=config.data.chat_template,
            max_length=config.data.max_length,
            min_loss_tokens=min_loss_tokens,
            media_root=media_root,
            media_uri_map=media_uri_map,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config.model.target_model_name_or_path,
        )
        train_collator = ConversationCollator(
            tokenizer=tokenizer,
            chat_template=config.data.chat_template,
            max_length=config.data.max_length,
            min_loss_tokens=min_loss_tokens,
        )
    target_model = load_target_cache_model(
        config.model.target_model_name_or_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to(device=device).eval()
    target_model.requires_grad_(False)
    target_hidden_size = get_target_hidden_size(target_model)
    if cli_args.fsdp:
        target_model = _wrap_target_model_with_fsdp(
            target_model=target_model,
            device=device,
            process_group=topology.fsdp_group,
            layerwise_only=context_parallel_size > 1,
        ).eval()
    dataloader = DataLoader(
        local_subset,
        batch_size=int(cli_args.local_batch_size),
        collate_fn=train_collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=int(cli_args.local_batch_size) * 4,
    )

    processed_local_samples = 0
    last_progress_printed = 0
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                processed_local_samples = min(
                    (batch_idx + 1) * int(cli_args.local_batch_size),
                    local_total_samples,
                )
                is_padding_step = cli_args.fsdp and batch_idx >= local_total_samples
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                should_write_batch = batch is not None and not is_padding_step
                if batch is None and not cli_args.fsdp:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                if batch is None:
                    batch = _build_dummy_model_inputs(device=device)
                batch = {
                    key: (
                        value.to(device, non_blocking=True)
                        if isinstance(value, torch.Tensor)
                        else value
                    )
                    for key, value in batch.items()
                }
                model_inputs = {
                    key: value for key, value in batch.items() if key != "loss_mask"
                }
                if context_parallel_size > 1:
                    if int(cli_args.local_batch_size) != 1:
                        raise ValueError(
                            "Target CP currently requires --local-batch-size 1."
                        )
                    full_embeddings, full_position_ids = (
                        _prepare_cp_embeddings_and_positions(
                            target_model=target_model,
                            model_inputs=model_inputs,
                        )
                    )
                    cp_attention_mask = model_inputs["attention_mask"].clone()
                    # Vision tensors are no longer needed after multimodal
                    # embeddings/M-RoPE positions have been constructed.
                    del model_inputs
                    for key in tuple(batch):
                        if key not in ("input_ids", "attention_mask", "loss_mask"):
                            del batch[key]
                    torch.cuda.empty_cache()
                    assert context_parallel_mesh is not None
                    target_result = _run_target_forward_native_cp(
                        target_model=target_model,
                        full_embeddings=full_embeddings,
                        full_position_ids=full_position_ids,
                        attention_mask=cp_attention_mask,
                        target_layer_ids=target_layer_ids,
                        topology=topology,
                        context_parallel_mesh=context_parallel_mesh,
                        device=device,
                    )
                else:
                    target_result = run_target_forward_with_hooks(
                        target_model=target_model,
                        model_inputs=model_inputs,
                        target_layer_ids=target_layer_ids,
                    )
                if not should_write_batch:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                assert target_result is not None
                for sample_idx_in_batch in range(batch["input_ids"].shape[0]):
                    valid_tokens = batch["attention_mask"][sample_idx_in_batch].bool()
                    if context_parallel_size > 1:
                        local_target_hidden_states = (
                            target_result.target_hidden_states[sample_idx_in_batch]
                        )
                        local_target_last_hidden_states = (
                            target_result.target_last_hidden_states[
                                sample_idx_in_batch
                            ]
                            if stores_target_last_hidden_states
                            else None
                        )
                    else:
                        output_valid_tokens = valid_tokens.to(
                            target_result.target_hidden_states.device
                        )
                        local_target_hidden_states = (
                            target_result.target_hidden_states[sample_idx_in_batch][
                                output_valid_tokens
                            ]
                        )
                        local_target_last_hidden_states = (
                            target_result.target_last_hidden_states[
                                sample_idx_in_batch
                            ][output_valid_tokens]
                            if stores_target_last_hidden_states
                            else None
                        )
                    writer.write_sample(
                        input_ids=batch["input_ids"][sample_idx_in_batch][valid_tokens],
                        attention_mask=batch["attention_mask"][sample_idx_in_batch][
                            valid_tokens
                        ],
                        loss_mask=batch["loss_mask"][sample_idx_in_batch][valid_tokens],
                        target_hidden_states=local_target_hidden_states,
                        target_last_hidden_states=local_target_last_hidden_states,
                    )
                if should_print_progress:
                    _print_prepare_progress(
                        global_rank=global_rank,
                        processed_samples=processed_local_samples,
                        total_samples=local_total_samples,
                    )
                    last_progress_printed = processed_local_samples
    finally:
        writer.close()
    del target_model
    torch.cuda.empty_cache()
    dataset.close()
    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        context_parallel_rank=topology.context_parallel_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    shard_map = None
    summaries = None
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
    broadcast_payload = [shard_map]
    dist.broadcast_object_list(broadcast_payload, src=0)
    shard_map = broadcast_payload[0]
    local_summary = load_local_cache_write_summary(rank_dir)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=local_summary,
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None
        num_valid_samples, index_files = finalize_target_cache_indices(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
            context_parallel_size=context_parallel_size,
        )
        _write_manifest(
            output_dir=output_dir,
            num_samples=num_valid_samples,
            index_files=index_files,
            config=config,
            train_data_paths=train_data_paths,
            target_layer_ids=target_layer_ids,
            hidden_size=target_hidden_size,
            min_loss_tokens=min_loss_tokens,
            multimodal=multimodal,
            processor_class=(type(processor).__name__ if processor is not None else None),
            media_root=media_root,
            media_uri_map=media_uri_map,
            context_parallel_size=context_parallel_size,
            fsdp_size=fsdp_size,
            target_micro_chunk_size=target_micro_chunk_size,
            target_cache_cpu_offload=target_cache_cpu_offload,
            stores_target_last_hidden_states=stores_target_last_hidden_states,
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{len(dataset)} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
