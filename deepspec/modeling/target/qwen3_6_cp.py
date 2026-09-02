"""Qwen3.6 target-model Context Parallel support.

Full-attention K/V shards are rotated by PyTorch native causal Context
Parallel. Qwen3.6's compact DeltaNet recurrent state is passed in causal
head/tail order, so no rank ever materializes the full sequence K/V cache.
"""

from contextlib import contextmanager
from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor.experimental import (
    context_parallel as torch_context_parallel,
)
from torch.distributed.tensor.experimental._context_parallel import (
    set_rotate_method,
)
from transformers import DynamicCache

from deepspec.modeling.target.common import TargetForwardResult
from deepspec.modeling.target_adapter import (
    get_decoder_layers,
    get_final_norm,
)


_CACHE_DTYPES = {
    torch.bfloat16: 0,
    torch.float32: 1,
    torch.float16: 2,
}
_CACHE_DTYPES_BY_CODE = {value: key for key, value in _CACHE_DTYPES.items()}
_CACHE_HEADER_SIZE = 11
_MAX_CACHE_NDIM = _CACHE_HEADER_SIZE - 3


def _get_hook_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        if isinstance(output[0], torch.Tensor):
            return output[0]
    raise TypeError(f"Unsupported target hook output type: {type(output)!r}")


def _capture_target_forward(
    *,
    target_model,
    model_inputs,
    target_layer_ids,
):
    layer_modules = get_decoder_layers(target_model)
    final_norm = get_final_norm(target_model)
    target_layer_ids = [int(layer_id) for layer_id in target_layer_ids]
    captured = {}
    final_hidden = []
    handles = []

    def capture_layer(layer_id):
        def hook(_module, _inputs, output):
            captured[layer_id] = _get_hook_tensor(output).detach()

        return hook

    def capture_decoder_input(_module, inputs):
        if not inputs:
            raise ValueError("Decoder pre-hook did not receive hidden states.")
        captured[-1] = _get_hook_tensor(inputs).detach()

    def capture_final(_module, _inputs, output):
        final_hidden.append(_get_hook_tensor(output).detach())

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
                layer_modules[layer_id].register_forward_hook(
                    capture_layer(layer_id)
                )
            )
        handles.append(final_norm.register_forward_hook(capture_final))
        with torch.no_grad():
            output = target_model(
                **model_inputs,
                output_hidden_states=False,
                use_cache=True,
                return_dict=True,
            )
        missing = [
            layer_id for layer_id in target_layer_ids if layer_id not in captured
        ]
        if missing:
            raise RuntimeError(f"Failed to capture target layers: {missing}.")
        last_hidden = (
            final_hidden[-1]
            if final_hidden
            else output.last_hidden_state.detach()
        )
        return TargetForwardResult(
            target_hidden_states=torch.cat(
                [captured[layer_id] for layer_id in target_layer_ids],
                dim=-1,
            ),
            target_last_hidden_states=last_hidden,
            context_start=0,
        )
    finally:
        for handle in handles:
            handle.remove()
        captured.clear()
        final_hidden.clear()


def _prepare_embeddings_and_positions(*, target_model, model_inputs):
    """Run Qwen's multimodal front end before sequence sharding."""

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
            inputs_embeds.device,
            inputs_embeds.dtype,
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
            inputs_embeds.device,
            inputs_embeds.dtype,
        )
        _, video_mask = target_model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            video_features=video_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    position_builder = getattr(target_model, "compute_3d_position_ids", None)
    if position_builder is not None:
        position_ids = position_builder(
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
    if position_ids is None:
        position_ids = torch.arange(
            input_ids.shape[1],
            dtype=torch.long,
            device=input_ids.device,
        ).unsqueeze(0).expand(input_ids.shape[0], -1)
    return inputs_embeds, position_ids


class _NativeContextParallelCache(DynamicCache):
    """Retain compact DeltaNet state but not native-CP full-attention K/V."""

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


def _pad_native_inputs(
    *,
    inputs_embeds,
    position_ids,
    attention_mask,
    context_parallel_size,
):
    sequence_length = int(inputs_embeds.shape[1])
    if attention_mask is None:
        attention_mask = torch.ones(
            (inputs_embeds.shape[0], sequence_length),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
    if int(attention_mask.shape[-1]) != sequence_length:
        raise ValueError(
            "Qwen3.6 target CP requires attention_mask to match inputs_embeds."
        )
    if not bool(torch.all(attention_mask != 0).item()):
        raise ValueError(
            "Qwen3.6 target CP requires one compact, unpadded sequence."
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
    )


def _global_group_rank(group, group_rank):
    if hasattr(dist, "get_global_rank"):
        return int(dist.get_global_rank(group, int(group_rank)))
    return int(dist.get_process_group_ranks(group)[int(group_rank)])


def _send_optional_tensor(*, tensor, dst, group, device):
    header = torch.zeros(
        _CACHE_HEADER_SIZE,
        dtype=torch.int64,
        device=device,
    )
    if tensor is not None:
        if tensor.dtype not in _CACHE_DTYPES:
            raise TypeError(f"Unsupported CP cache dtype: {tensor.dtype}.")
        if tensor.ndim > _MAX_CACHE_NDIM:
            raise ValueError(
                f"CP cache tensor rank {tensor.ndim} exceeds {_MAX_CACHE_NDIM}."
            )
        header[0] = 1
        header[1] = _CACHE_DTYPES[tensor.dtype]
        header[2] = tensor.ndim
        header[3 : 3 + tensor.ndim] = torch.tensor(
            tensor.shape,
            dtype=torch.int64,
            device=device,
        )
    dist.send(header, dst=dst, group=group)
    if tensor is not None:
        dist.send(tensor.contiguous(), dst=dst, group=group)


def _recv_optional_tensor(*, src, group, device):
    header = torch.empty(
        _CACHE_HEADER_SIZE,
        dtype=torch.int64,
        device=device,
    )
    dist.recv(header, src=src, group=group)
    header_cpu = header.cpu()
    if int(header_cpu[0].item()) == 0:
        return None
    dtype_code = int(header_cpu[1].item())
    ndim = int(header_cpu[2].item())
    if not 0 < ndim <= _MAX_CACHE_NDIM:
        raise RuntimeError(f"Received invalid CP cache tensor rank {ndim}.")
    shape = tuple(int(value) for value in header_cpu[3 : 3 + ndim].tolist())
    tensor = torch.empty(
        shape,
        dtype=_CACHE_DTYPES_BY_CODE[dtype_code],
        device=device,
    )
    dist.recv(tensor, src=src, group=group)
    return tensor


def _send_linear_state(*, cache, layer_idx, dst, group, device):
    layer = cache.layers[int(layer_idx)]
    if not hasattr(layer, "is_conv_states_initialized"):
        raise TypeError(
            f"Target cache layer {layer_idx} is not a DeltaNet cache."
        )
    if isinstance(layer.conv_states, dict):
        state_indices = tuple(range(int(layer.number_of_states)))
        for state_idx in state_indices:
            _send_optional_tensor(
                tensor=(
                    layer.conv_states[state_idx]
                    if layer.is_conv_states_initialized[state_idx]
                    else None
                ),
                dst=dst,
                group=group,
                device=device,
            )
            _send_optional_tensor(
                tensor=(
                    layer.recurrent_states[state_idx]
                    if layer.is_recurrent_states_initialized[state_idx]
                    else None
                ),
                dst=dst,
                group=group,
                device=device,
            )
        return
    # Compatibility with Transformers releases that stored one state directly.
    _send_optional_tensor(
        tensor=layer.conv_states if layer.is_conv_states_initialized else None,
        dst=dst,
        group=group,
        device=device,
    )
    _send_optional_tensor(
        tensor=(
            layer.recurrent_states
            if layer.is_recurrent_states_initialized
            else None
        ),
        dst=dst,
        group=group,
        device=device,
    )


def _recv_linear_state(*, cache, layer_idx, src, group, device):
    layer = cache.layers[int(layer_idx)]
    state_indices = (
        tuple(range(int(layer.number_of_states)))
        if isinstance(layer.conv_states, dict)
        else (None,)
    )
    for state_idx in state_indices:
        conv_states = _recv_optional_tensor(
            src=src,
            group=group,
            device=device,
        )
        recurrent_states = _recv_optional_tensor(
            src=src,
            group=group,
            device=device,
        )
        if conv_states is None or recurrent_states is None:
            raise RuntimeError(
                f"Missing Qwen3.6 DeltaNet state for layer {layer_idx}, "
                f"state {state_idx if state_idx is not None else 0}."
            )
        if state_idx is None:
            if not layer.is_conv_states_initialized:
                layer.lazy_initialization(conv_states=conv_states)
            if not layer.is_recurrent_states_initialized:
                layer.lazy_initialization(recurrent_states=recurrent_states)
            local_conv_states = layer.conv_states
            local_recurrent_states = layer.recurrent_states
        else:
            if not layer.is_conv_states_initialized[state_idx]:
                layer.lazy_initialization(
                    conv_states=conv_states,
                    state_idx=state_idx,
                )
            if not layer.is_recurrent_states_initialized[state_idx]:
                layer.lazy_initialization(
                    recurrent_states=recurrent_states,
                    state_idx=state_idx,
                )
            local_conv_states = layer.conv_states[state_idx]
            local_recurrent_states = layer.recurrent_states[state_idx]
        if local_conv_states.shape != conv_states.shape:
            raise RuntimeError("Received DeltaNet conv state has the wrong shape.")
        if local_recurrent_states.shape != recurrent_states.shape:
            raise RuntimeError(
                "Received DeltaNet recurrent state has the wrong shape."
            )
        local_conv_states.copy_(conv_states)
        local_recurrent_states.copy_(recurrent_states)
        if state_idx is None:
            layer.has_previous_state = True
        else:
            layer.has_previous_state[state_idx] = True


def _split_linear_mask(attention_mask, split_size):
    if attention_mask is None:
        return None, None
    if attention_mask.ndim != 2 or int(attention_mask.shape[-1]) != 2 * split_size:
        raise ValueError(
            "Qwen3.6 target CP expects a two-part local attention mask."
        )
    return (
        attention_mask[:, :split_size],
        attention_mask[:, split_size:],
    )


def _run_linear_attention_cp(
    *,
    original_forward,
    layer_idx,
    hidden_states,
    cache_params,
    attention_mask,
    context_parallel_group,
    context_parallel_rank,
    context_parallel_size,
    device,
    forward_kwargs,
):
    local_length = int(hidden_states.shape[1])
    if local_length % 2 != 0:
        raise ValueError(
            "Native causal CP requires equal local head/tail chunks."
        )
    if not isinstance(cache_params, _NativeContextParallelCache):
        raise TypeError(
            "Qwen3.6 target CP requires _NativeContextParallelCache."
        )
    split_size = local_length // 2
    head_mask, tail_mask = _split_linear_mask(attention_mask, split_size)
    previous_rank = (
        _global_group_rank(
            context_parallel_group,
            context_parallel_rank - 1,
        )
        if context_parallel_rank > 0
        else None
    )
    next_rank = (
        _global_group_rank(
            context_parallel_group,
            context_parallel_rank + 1,
        )
        if context_parallel_rank < context_parallel_size - 1
        else None
    )

    if previous_rank is not None:
        _recv_linear_state(
            cache=cache_params,
            layer_idx=layer_idx,
            src=previous_rank,
            group=context_parallel_group,
            device=device,
        )
    head_output = original_forward(
        hidden_states=hidden_states[:, :split_size],
        cache_params=cache_params,
        attention_mask=head_mask,
        **forward_kwargs,
    )
    if next_rank is not None:
        _send_linear_state(
            cache=cache_params,
            layer_idx=layer_idx,
            dst=next_rank,
            group=context_parallel_group,
            device=device,
        )

    if next_rank is not None:
        _recv_linear_state(
            cache=cache_params,
            layer_idx=layer_idx,
            src=next_rank,
            group=context_parallel_group,
            device=device,
        )
    tail_output = original_forward(
        hidden_states=hidden_states[:, split_size:],
        cache_params=cache_params,
        attention_mask=tail_mask,
        **forward_kwargs,
    )
    if previous_rank is not None:
        _send_linear_state(
            cache=cache_params,
            layer_idx=layer_idx,
            dst=previous_rank,
            group=context_parallel_group,
            device=device,
        )
    return torch.cat([head_output, tail_output], dim=1)


@contextmanager
def _patch_linear_attention(
    *,
    target_model,
    context_parallel_group,
    context_parallel_rank,
    context_parallel_size,
    device,
):
    patched = []
    for decoder_layer in get_decoder_layers(target_model):
        unwrapped = (
            decoder_layer.module
            if isinstance(decoder_layer, FSDP)
            else decoder_layer
        )
        linear_attention = getattr(unwrapped, "linear_attn", None)
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
            return _run_linear_attention_cp(
                original_forward=_original_forward,
                layer_idx=_layer_idx,
                hidden_states=hidden_states,
                cache_params=cache_params,
                attention_mask=attention_mask,
                context_parallel_group=context_parallel_group,
                context_parallel_rank=context_parallel_rank,
                context_parallel_size=context_parallel_size,
                device=device,
                forward_kwargs=kwargs,
            )

        linear_attention.forward = cp_forward
        patched.append((linear_attention, original_forward))
    try:
        yield
    finally:
        for linear_attention, original_forward in patched:
            linear_attention.forward = original_forward


def _qwen3_6_forward_context_parallel(
    self,
    *,
    model_inputs,
    target_layer_ids,
    context_parallel_group,
    context_parallel_rank,
    context_parallel_size,
    device,
    **_parallel_kwargs,
):
    del _parallel_kwargs
    if int(context_parallel_size) <= 1:
        raise ValueError("Qwen3.6 native target CP requires CP size > 1.")
    inputs_embeds, position_ids = _prepare_embeddings_and_positions(
        target_model=self,
        model_inputs=model_inputs,
    )
    # The vision tensors are no longer needed once visual features and M-RoPE
    # positions have been folded into the language-model inputs. The caller
    # deliberately passes its batch dictionary so these references can be
    # released before the long-context decoder starts.
    for key in tuple(model_inputs):
        if key not in ("input_ids", "attention_mask"):
            del model_inputs[key]
    torch.cuda.empty_cache()
    inputs_embeds, position_ids, attention_mask = _pad_native_inputs(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        attention_mask=model_inputs.get("attention_mask"),
        context_parallel_size=context_parallel_size,
    )
    buffers = [inputs_embeds, position_ids, attention_mask]
    buffer_seq_dims = [1, position_ids.ndim - 1, 1]
    target_cache = _NativeContextParallelCache(config=self.config)
    context_mesh = DeviceMesh.from_group(
        context_parallel_group,
        "cuda",
        mesh_dim_names=("context",),
    )

    with torch_context_parallel(
        context_mesh,
        buffers=buffers,
        buffer_seq_dims=buffer_seq_dims,
        no_restore_buffers=set(buffers),
    ):
        for buffer in buffers:
            buffer.set_(buffer.clone())
        with _patch_linear_attention(
            target_model=self,
            context_parallel_group=context_parallel_group,
            context_parallel_rank=int(context_parallel_rank),
            context_parallel_size=int(context_parallel_size),
            device=device,
        ):
            result = _capture_target_forward(
                target_model=self,
                model_inputs={
                    "inputs_embeds": inputs_embeds,
                    "position_ids": position_ids,
                    "attention_mask": attention_mask,
                    "past_key_values": target_cache,
                },
                target_layer_ids=target_layer_ids,
            )
    del target_cache
    return result


def install_qwen3_6_ring_context_parallel(target_model):
    model_type = str(target_model.config.model_type).lower()
    if model_type not in ("qwen3_5", "qwen3_5_text"):
        raise ValueError(
            "Qwen3.6 CP adapter received model_type="
            f"{target_model.config.model_type!r}."
        )
    if getattr(target_model, "_deepspec_qwen3_6_cp_installed", False):
        return target_model
    set_rotate_method("alltoall")
    target_model.forward_context_parallel = MethodType(
        _qwen3_6_forward_context_parallel,
        target_model,
    )
    target_model._deepspec_context_layout = "native_head_tail"
    target_model._deepspec_qwen3_6_cp_installed = True
    return target_model


__all__ = ["install_qwen3_6_ring_context_parallel"]
