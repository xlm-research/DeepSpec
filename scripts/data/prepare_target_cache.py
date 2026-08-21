import argparse
import json
import os

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from deepspec.data import (
    ConversationCollator,
    MultimodalConversationCollator,
)
from deepspec.data.parser import parse_media_uri_map_entries
from deepspec.data.jsonl_dataset import JsonLineDataset
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
from deepspec.modeling.target import (
    TargetForwardResult,
    install_target_context_parallel,
)
from deepspec.modeling.target_adapter import (
    get_final_norm,
    get_language_backbone,
    get_target_hidden_size,
    is_multimodal_config,
)
from deepspec.modeling.deepseek_v4_parallel import (
    parallelize_deepseek_v4_model,
)
from deepspec.utils import (
    CustomJSONEncoder,
    build_parallel_topology,
    compute_context_parallel_range,
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
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
    FineGrainedFP8Config,
)

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# PyTorch 2.10 Inductor still reads the legacy allow_tf32 flag while compiling.
torch.set_float32_matmul_precision("high")


def _get_target_backbone(target_model):
    return get_language_backbone(target_model)


def _get_target_hidden_size(target_model) -> int:
    return get_target_hidden_size(target_model)


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
):
    backbone = _get_target_backbone(target_model)
    layer_modules = backbone.layers
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
            raise ValueError("Decoder pre-hook did not receive hidden states.")
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
        handles.append(
            final_norm.register_forward_hook(capture_last_hidden_state)
        )

        with torch.no_grad():
            target_output = target_model(
                **model_inputs,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
            target_last_hidden_states = (
                captured_last_hidden_state[-1]
                if captured_last_hidden_state
                else target_output.last_hidden_state.detach()
            )
            missing = [
                layer_id
                for layer_id in target_layer_ids
                if layer_id not in captured_hidden_states
            ]
            if missing:
                raise RuntimeError(f"Failed to capture target layers: {missing}.")
            target_hidden_states = torch.cat(
                [captured_hidden_states[layer_id] for layer_id in target_layer_ids],
                dim=-1,
            )
    finally:
        for handle in handles:
            handle.remove()
        captured_hidden_states.clear()
        captured_last_hidden_state.clear()

    return TargetForwardResult(
        target_hidden_states=target_hidden_states,
        target_last_hidden_states=target_last_hidden_states,
        context_start=0,
    )


def _dequantizing_config(target_config):
    """Return an FP8 loading config that materializes ordinary BF16 weights."""

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
    """FSDP1 requires a uniform floating-point dtype within each flat shard."""

    for parameter in target_model.parameters():
        if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
            parameter.data = parameter.data.to(dtype=torch.bfloat16)


def _build_fsdp_target_model(
    *, model_name_or_path: str, target_config, fsdp_rank: int
):
    """Load one CPU checkpoint per FSDP group and use meta tensors elsewhere.

    DeepSeek-V4-Flash is roughly 149 GB in its source FP8 checkpoint and about
    300 GB after BF16 dequantization.  Loading it independently in every local
    process would multiply host memory by the number of GPUs before FSDP even
    starts.  Only FSDP group rank zero reads/dequantizes the checkpoint; the
    other ranks construct the identical module graph on ``meta``.  Layer-wise
    FSDP then materializes, broadcasts, and shards one decoder block at a time.
    """

    if int(fsdp_rank) == 0:
        quantization_config = _dequantizing_config(target_config)
        load_kwargs = {
            "config": target_config,
            "dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if str(target_config.model_type).lower() in (
            "qwen3_5",
            "qwen3_5_text",
        ):
            load_kwargs["attn_implementation"] = "sdpa"
        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
        target_model = AutoModel.from_pretrained(
            model_name_or_path,
            **load_kwargs,
        )
        _normalize_target_parameter_dtype(target_model)
        return target_model

    with init_empty_weights(include_buffers=True):
        config_kwargs = {"dtype": torch.bfloat16}
        if str(target_config.model_type).lower() in (
            "qwen3_5",
            "qwen3_5_text",
        ):
            config_kwargs["attn_implementation"] = "sdpa"
        target_model = AutoModel.from_config(target_config, **config_kwargs)
    # Some custom modules allocate parameters on the current CUDA device even
    # inside init_empty_weights(). FSDP rejects a mixed CUDA/meta layer before
    # invoking param_init_fn, so normalize every non-source tensor to meta.
    target_model.to_empty(device=torch.device("meta"), recurse=True)
    _normalize_target_parameter_dtype(target_model)
    return target_model


def _materialize_tensor_on_device(
    *, module: nn.Module, tensor_name: str, device, is_parameter: bool
) -> torch.Tensor:
    """Move a real tensor or allocate its meta peer while preserving metadata."""

    registry = module._parameters if is_parameter else module._buffers
    tensor = registry[tensor_name]
    if tensor is None:
        raise RuntimeError(f"Cannot materialize missing tensor {tensor_name!r}.")
    if tensor.is_meta:
        materialized = torch.empty_like(tensor, device=device)
    else:
        materialized = tensor.to(device=device)
    if is_parameter:
        materialized = nn.Parameter(
            materialized,
            requires_grad=bool(tensor.requires_grad),
        )
    registry[tensor_name] = materialized
    return materialized


def _materialize_and_sync_replicated_target_state(
    *, target_model, decoder_layers, device, process_group
) -> None:
    """Replicate only embedding/final-state tensors across an FSDP group."""

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
    source_rank = (
        int(dist.get_global_rank(process_group, 0))
        if hasattr(dist, "get_global_rank")
        else int(dist.get_process_group_ranks(process_group)[0])
    )
    for qualified_name, parameter in list(target_model.named_parameters()):
        if id(parameter) in decoder_parameter_ids:
            continue
        module_name, _, tensor_name = qualified_name.rpartition(".")
        module = (
            target_model.get_submodule(module_name)
            if module_name
            else target_model
        )
        materialized = _materialize_tensor_on_device(
            module=module,
            tensor_name=tensor_name,
            device=device,
            is_parameter=True,
        )
        dist.broadcast(materialized.data, src=source_rank, group=process_group)
    for qualified_name, buffer in list(target_model.named_buffers()):
        if id(buffer) in decoder_buffer_ids:
            continue
        module_name, _, tensor_name = qualified_name.rpartition(".")
        module = (
            target_model.get_submodule(module_name)
            if module_name
            else target_model
        )
        materialized = _materialize_tensor_on_device(
            module=module,
            tensor_name=tensor_name,
            device=device,
            is_parameter=False,
        )
        dist.broadcast(materialized, src=source_rank, group=process_group)


def _materialize_meta_module(module: nn.Module, *, device) -> None:
    if any(
        tensor is not None and tensor.is_meta
        for tensor in (
            *module.parameters(recurse=False),
            *module.buffers(recurse=False),
        )
    ):
        module.to_empty(device=device, recurse=False)


def _wrap_target_model_with_fsdp(*, target_model, device, process_group):
    """FULL_SHARD the target model one decoder layer at a time.

    Layer-wise units bound the transient all-gather to one target layer.  The
    model is wrapped while still on CPU, so the full checkpoint is never moved
    to a single GPU before sharding.
    """

    backbone = _get_target_backbone(target_model)
    decoder_layers = list(backbone.layers)
    if not decoder_layers:
        raise ValueError("Target model does not expose decoder layers for FSDP.")
    _materialize_and_sync_replicated_target_state(
        target_model=target_model,
        decoder_layers=decoder_layers,
        device=device,
        process_group=process_group,
    )
    fsdp_kwargs = {
        "process_group": process_group,
        "device_id": device,
        "limit_all_gathers": True,
        "mixed_precision": MixedPrecision(
            param_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "sync_module_states": True,
        "use_orig_params": True,
        "param_init_fn": lambda module: _materialize_meta_module(
            module, device=device
        ),
    }
    # Keep the model root unwrapped so a model-native CP entry point can drive
    # the decoder.  Only decoder blocks contain the large trainable parameter
    # sets; wrapping them individually bounds each parameter all-gather to one
    # layer while embeddings and final normalization stay replicated.
    for layer_idx, decoder_layer in enumerate(decoder_layers):
        backbone.layers[layer_idx] = FSDP(decoder_layer, **fsdp_kwargs)
    return target_model


def _run_target_forward_context_parallel(
    *,
    target_model,
    model_inputs,
    target_layer_ids,
    topology,
    device,
):
    """Dispatch to a model family's exact ring-context implementation.

    The parallel plumbing intentionally does not reconstruct the full sequence
    or gather hidden states on a leader. A model adapter returns only the token
    shard consumed by the matching draft CP rank.
    """

    cp_forward = getattr(target_model, "forward_context_parallel", None)
    if cp_forward is None:
        model_type = str(target_model.config.model_type)
        raise NotImplementedError(
            f"Target model type {model_type!r} does not expose "
            "forward_context_parallel(). Add that model-specific ring "
            "attention adapter before running with --context-parallel-size > 1."
        )
    # Native CP may shard aligned input buffers in place. Retain the global
    # mask for validation and for the cache writer that follows this call.
    global_attention_mask = model_inputs["attention_mask"].clone()
    sequence_length = int(global_attention_mask.sum(dim=1)[0].item())
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
    model_inputs["attention_mask"] = global_attention_mask
    if not isinstance(result, TargetForwardResult):
        raise TypeError(
            "forward_context_parallel() must return TargetForwardResult, got "
            f"{type(result)!r}."
        )
    local_length = int(result.target_hidden_states.shape[1])
    context_layout = str(
        getattr(target_model, "_deepspec_context_layout", "contiguous")
    )
    if context_layout == "contiguous":
        expected_start, expected_end = compute_context_parallel_range(
            sequence_length=sequence_length,
            context_parallel_rank=topology.context_parallel_rank,
            context_parallel_size=topology.context_parallel_size,
        )
        if (
            result.context_start != expected_start
            or local_length != expected_end - expected_start
        ):
            raise RuntimeError(
                "Target CP adapter returned the wrong contiguous shard: "
                f"expected [{expected_start}, {expected_end}), got "
                f"start={result.context_start}, length={local_length}."
            )
    elif context_layout == "native_head_tail":
        alignment = 2 * int(topology.context_parallel_size)
        padded_length = (
            (sequence_length + alignment - 1) // alignment
        ) * alignment
        expected_length = padded_length // int(topology.context_parallel_size)
        if result.context_start != 0 or local_length != expected_length:
            raise RuntimeError(
                "Qwen3.6 target CP returned the wrong native head/tail shard: "
                f"expected start=0, length={expected_length}; got "
                f"start={result.context_start}, length={local_length}."
            )
    else:
        raise RuntimeError(
            f"Target CP adapter declared unsupported layout {context_layout!r}."
        )
    if int(result.target_last_hidden_states.shape[1]) != local_length:
        raise RuntimeError("Target CP hidden-state tensors have different lengths.")
    return result


def _build_dummy_batch(*, device, sequence_length: int = 1):
    """Return one valid token so padded FSDP steps enter every layer."""

    sequence_length = max(int(sequence_length), 1)
    return {
        "input_ids": torch.zeros(
            (1, sequence_length), dtype=torch.long, device=device
        ),
        "attention_mask": torch.ones(
            (1, sequence_length), dtype=torch.long, device=device
        ),
        "loss_mask": torch.zeros(
            (1, sequence_length), dtype=torch.long, device=device
        ),
    }


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
        help="Shard target-model parameters with layer-wise FSDP FULL_SHARD.",
    )
    parser.add_argument(
        "--fsdp-size",
        type=int,
        default=None,
        help=(
            "Ranks per target FSDP group. Defaults to train.fsdp_size, then "
            "world_size / (context_parallel_size * expert_parallel_size * "
            "tensor_parallel_size)."
        ),
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=None,
        help=(
            "Ranks cooperating on one sequence with model-native ring CP. "
            "Defaults to train.context_parallel_size, then 1."
        ),
    )
    parser.add_argument(
        "--expert-parallel-size",
        type=int,
        default=None,
        help=(
            "Shard DeepSeek-V4 routed experts across this many ranks. "
            "Defaults to train.expert_parallel_size, then 1."
        ),
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=None,
        help=(
            "Shard DeepSeek-V4 attention heads, expert intermediate channels, "
            "and vocabulary tensors across this many ranks. Defaults to "
            "train.tensor_parallel_size, then 1."
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
            "Rewrite image/video URI prefixes before loading media. Repeat "
            "for multiple mappings; the longest matching prefix wins."
        ),
    )
    cli_args = parser.parse_args()
    try:
        cli_args.media_uri_map = parse_media_uri_map_entries(
            cli_args.media_uri_map
        )
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
    context_layout: str,
    expert_parallel_size: int,
    tensor_parallel_size: int,
    fsdp_size: int,
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
            "cache_context_parallel_size": int(context_parallel_size),
            "context_layout": str(context_layout),
            "index_files": [str(file_name) for file_name in index_files],
            "target_context_parallel_implementation": (
                "model_native_ring" if context_parallel_size > 1 else "disabled"
            ),
            "target_fsdp_size": int(fsdp_size),
            "stores_target_last_hidden_states": bool(
                stores_target_last_hidden_states
            ),
            "target_expert_parallel_size": int(expert_parallel_size),
            "target_expert_parallel_implementation": (
                "token_all_to_all" if expert_parallel_size > 1 else "disabled"
            ),
            "target_tensor_parallel_size": int(tensor_parallel_size),
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


def _print_prepare_progress(
    *, global_rank: int, processed_samples: int, total_samples: int
):
    print(
        f"[prepare rank {global_rank}] {processed_samples}/{total_samples} samples",
        flush=True,
    )


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = list(cli_args.train_data_path)
    target_layer_ids = [int(layer_id) for layer_id in config.model.target_layer_ids]
    min_loss_tokens = int(cli_args.min_loss_tokens)
    stores_target_last_hidden_states = bool(
        config.data.get("store_target_last_hidden_states", True)
    )
    target_config = AutoConfig.from_pretrained(
        config.model.target_model_name_or_path,
    )
    configured_multimodal = config.data.get("multimodal")
    multimodal = (
        (
            str(target_config.model_type).lower()
            in ("qwen3_5", "qwen3_5_text")
            and is_multimodal_config(target_config)
        )
        if configured_multimodal is None
        else bool(configured_multimodal)
    )
    media_root = (
        os.path.abspath(cli_args.media_root)
        if cli_args.media_root is not None
        else None
    )
    media_uri_map = cli_args.media_uri_map
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist(local_rank)
    configured_context_parallel_size = cli_args.context_parallel_size
    if configured_context_parallel_size is None:
        configured_context_parallel_size = config.train.get(
            "context_parallel_size", 1
        )
    context_parallel_size = int(configured_context_parallel_size)
    if context_parallel_size < 1:
        raise ValueError("--context-parallel-size must be positive.")
    configured_expert_parallel_size = cli_args.expert_parallel_size
    if configured_expert_parallel_size is None:
        configured_expert_parallel_size = config.train.get(
            "expert_parallel_size", 1
        )
    expert_parallel_size = int(configured_expert_parallel_size)
    configured_tensor_parallel_size = cli_args.tensor_parallel_size
    if configured_tensor_parallel_size is None:
        configured_tensor_parallel_size = config.train.get(
            "tensor_parallel_size", 1
        )
    tensor_parallel_size = int(configured_tensor_parallel_size)
    if expert_parallel_size < 1 or tensor_parallel_size < 1:
        raise ValueError("EP and TP sizes must both be positive.")
    if str(target_config.model_type).lower() in ("qwen3_5", "qwen3_5_text") and (
        expert_parallel_size > 1 or tensor_parallel_size > 1
    ):
        raise ValueError(
            "Qwen3.6 target-cache generation supports CP and FSDP; set "
            "expert_parallel_size=1 and tensor_parallel_size=1."
        )
    if context_parallel_size > 1 and not cli_args.fsdp:
        raise ValueError("Target Context Parallel currently requires --fsdp.")
    if cli_args.fsdp:
        configured_fsdp_size = cli_args.fsdp_size
        if configured_fsdp_size is None:
            configured_fsdp_size = config.train.get("fsdp_size")
        fsdp_size = (
            world_size
            // (
                context_parallel_size
                * expert_parallel_size
                * tensor_parallel_size
            )
            if configured_fsdp_size is None
            else int(configured_fsdp_size)
        )
    else:
        if cli_args.fsdp_size is not None:
            raise ValueError("--fsdp-size requires --fsdp.")
        fsdp_size = 1
    topology = build_parallel_topology(
        context_parallel_size=context_parallel_size,
        expert_parallel_size=expert_parallel_size,
        tensor_parallel_size=tensor_parallel_size,
        fsdp_size=fsdp_size,
        create_fsdp_groups=bool(cli_args.fsdp),
    )
    local_batch_size = int(cli_args.local_batch_size)
    if (
        cli_args.fsdp
        or context_parallel_size > 1
        or expert_parallel_size > 1
        or tensor_parallel_size > 1
    ) and local_batch_size != 1:
        raise ValueError(
            "Target-cache FSDP/CP/EP/TP currently requires "
            "--local-batch-size 1."
        )
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
                "expert_parallel_size": expert_parallel_size,
                "tensor_parallel_size": tensor_parallel_size,
                "stores_target_last_hidden_states": (
                    stores_target_last_hidden_states
                ),
                "context_parallel_implementation": (
                    "model_native_ring"
                    if context_parallel_size > 1
                    else "disabled"
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
    dataset_size = len(dataset)

    local_start, local_end = compute_local_sample_range(
        num_samples=dataset_size,
        rank=topology.sample_parallel_rank,
        world_size=topology.sample_parallel_size,
    )
    local_total_samples = local_end - local_start
    local_indices = list(range(local_start, local_end))
    if cli_args.fsdp:
        # Ranks in an FSDP group may own different samples, but they must enter
        # the same number of wrapped-layer collectives.  Duplicate one local
        # index only as a non-writing padding step on shorter ranks.
        max_local_steps = (
            dataset_size + topology.sample_parallel_size - 1
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
    if cli_args.fsdp:
        target_model = _build_fsdp_target_model(
            model_name_or_path=config.model.target_model_name_or_path,
            target_config=target_config,
            fsdp_rank=topology.fsdp_rank,
        ).eval()
    else:
        load_kwargs = {
            "config": target_config,
            "dtype": torch.bfloat16,
        }
        if str(target_config.model_type).lower() in (
            "qwen3_5",
            "qwen3_5_text",
        ):
            load_kwargs["attn_implementation"] = "sdpa"
        target_model = AutoModel.from_pretrained(
            config.model.target_model_name_or_path,
            **load_kwargs,
        ).eval()
    target_model.requires_grad_(False)
    if expert_parallel_size > 1 or tensor_parallel_size > 1:
        target_model = parallelize_deepseek_v4_model(
            target_model,
            topology=topology,
            draft=False,
        )
    if context_parallel_size > 1:
        target_model = install_target_context_parallel(target_model)
    context_layout = str(
        getattr(target_model, "_deepspec_context_layout", "contiguous")
    )
    target_hidden_size = _get_target_hidden_size(target_model)
    if cli_args.fsdp:
        target_model = _wrap_target_model_with_fsdp(
            target_model=target_model,
            device=device,
            process_group=topology.fsdp_group,
        ).eval()
    else:
        target_model = target_model.to(device=device)
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
        context_layout=context_layout,
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
                should_print_progress = (
                    processed_local_samples - last_progress_printed >= 100
                    or processed_local_samples == local_total_samples
                )
                is_padding_step = bool(cli_args.fsdp) and (
                    batch_idx >= local_total_samples
                )
                should_write_batch = (
                    batch is not None
                    and not is_padding_step
                    and topology.expert_parallel_rank == 0
                    and topology.tensor_parallel_rank == 0
                )
                if batch is None and not cli_args.fsdp:
                    if should_print_progress:
                        _print_prepare_progress(
                            global_rank=global_rank,
                            processed_samples=processed_local_samples,
                            total_samples=local_total_samples,
                        )
                        last_progress_printed = processed_local_samples
                    continue
                if batch is None or is_padding_step:
                    batch = _build_dummy_batch(
                        device=device,
                        sequence_length=context_parallel_size,
                    )
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
                loss_mask = batch.pop("loss_mask")
                model_inputs = batch
                if context_parallel_size > 1:
                    if int(batch["input_ids"].shape[0]) != 1:
                        raise ValueError("Target CP only supports batch size 1.")
                    attention_mask = batch["attention_mask"]
                    sequence_length = int(attention_mask.sum(dim=1)[0].item())
                    if sequence_length < 1:
                        raise ValueError("Target CP received an empty sequence.")
                    if (
                        not bool(attention_mask[0, :sequence_length].all())
                        or bool(attention_mask[0, sequence_length:].any())
                    ):
                        raise ValueError("Target CP requires right-padded inputs.")
                    padded_length = int(batch["input_ids"].shape[1])
                    for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
                        tensor = batch.get(key)
                        if (
                            tensor is not None
                            and tensor.ndim >= 2
                            and int(tensor.shape[-1]) == padded_length
                        ):
                            batch[key] = tensor[..., :sequence_length]
                    loss_mask = loss_mask[..., :sequence_length]
                    target_result = _run_target_forward_context_parallel(
                        target_model=target_model,
                        model_inputs=model_inputs,
                        target_layer_ids=target_layer_ids,
                        topology=topology,
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
                    del target_result, model_inputs, batch, loss_mask
                    continue
                for sample_idx_in_batch in range(batch["input_ids"].shape[0]):
                    valid_tokens = batch["attention_mask"][sample_idx_in_batch].bool()
                    if context_parallel_size > 1:
                        if sample_idx_in_batch != 0:
                            raise RuntimeError("Target CP only supports batch size 1.")
                        local_hidden_states = target_result.target_hidden_states[0]
                        local_last_hidden_states = (
                            target_result.target_last_hidden_states[0]
                            if stores_target_last_hidden_states
                            else None
                        )
                        context_start = int(target_result.context_start)
                    else:
                        output_valid_tokens = valid_tokens.to(
                            target_result.target_hidden_states.device
                        )
                        local_hidden_states = target_result.target_hidden_states[
                            sample_idx_in_batch
                        ][output_valid_tokens]
                        local_last_hidden_states = (
                            target_result.target_last_hidden_states[
                                sample_idx_in_batch
                            ][output_valid_tokens]
                            if stores_target_last_hidden_states
                            else None
                        )
                        context_start = 0
                    writer.write_sample(
                        context_start=context_start,
                        input_ids=batch["input_ids"][sample_idx_in_batch][valid_tokens],
                        attention_mask=batch["attention_mask"][sample_idx_in_batch][
                            valid_tokens
                        ],
                        loss_mask=loss_mask[sample_idx_in_batch][valid_tokens],
                        target_hidden_states=local_hidden_states,
                        target_last_hidden_states=local_last_hidden_states,
                    )
                del target_result, model_inputs, batch, loss_mask
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
            context_layout=context_layout,
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
            processor_class=(
                type(processor).__name__ if processor is not None else None
            ),
            media_root=media_root,
            media_uri_map=media_uri_map,
            context_parallel_size=context_parallel_size,
            context_layout=context_layout,
            expert_parallel_size=expert_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            fsdp_size=fsdp_size,
            stores_target_last_hidden_states=(
                stores_target_last_hidden_states
            ),
            shards=shards,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared target cache at {output_dir} with "
            f"{num_valid_samples}/{dataset_size} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if os.path.exists(".git"):
        print("git status:", "\n\n".join(get_git_sha(detail_info=True)))
        print("git diff:", get_git_diff())
    torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
