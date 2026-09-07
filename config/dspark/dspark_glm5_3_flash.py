import os
import math
import sys

from deepspec.trainer import Glm5NextDSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec_glm5"
exp_name = "dspark_glm5_3_flash_128k"
seed = 42

# Torchrun exports the GPU-process world size before each worker imports this
# config. Keep the historical 8-GPU values when the file is inspected outside
# a distributed launch, while deriving node-local HSDP/EP defaults at runtime.
runtime_world_size = int(os.environ.get("WORLD_SIZE", "8"))
runtime_local_world_size = int(
    os.environ.get("LOCAL_WORLD_SIZE", str(runtime_world_size))
)
runtime_node_count = max(runtime_world_size // runtime_local_world_size, 1)
runtime_draft_ep = math.gcd(runtime_local_world_size, 288)
runtime_target_is_node_local = runtime_local_world_size % 4 == 0
runtime_target_dp_replicate = (
    runtime_node_count if runtime_target_is_node_local else 1
)
runtime_target_dp_shard = (
    runtime_local_world_size // 4
    if runtime_target_is_node_local
    else max(runtime_world_size // 4, 1)
)
runtime_target_ep = math.gcd(runtime_target_dp_shard * 4, 288)

model = dict(
    target_model_name_or_path=(
        "/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash"
    ),
    block_size=7,
    num_draft_layers=3,
    # Sample the end of the dense prefix plus middle/late sparse layers. This
    # preserves progressive target depth instead of feeding the draft three
    # nearly adjacent early representations. The full 45-layer target still
    # runs so L1/confidence supervision uses its true final normalized state.
    target_layer_ids=[2, 22, 42],
    # GLM-5.3 has no mask token. Use the final reserved vocabulary row.
    mask_token_id=154879,
    num_anchors=512,
    sliding_window=128,
    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Glm5NextDSparkTrainer,
    lr=1.0e-5,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=max(runtime_world_size, 8),
    # Requested optimizer-aligned target-cache partition count. With model
    # swap, partition the entire dataset and repeat its boundaries each epoch.
    data_batch_size=8,
    # Opt-in lifecycle that alternates the full GLM target with the complete
    # draft training state. The launcher uses data_batch_size by default, or
    # clears it when PARTITION_MAX_SAMPLES selects a per-partition sample cap.
    partitioned_model_swap=dict(
        enabled=False,
        max_samples=512,
        target_backend="native",
        # Independent node-local TP4 vLLM processes exit before draft loading.
        # Use TARGET_BACKEND=vllm with the FSDP launcher to enable this backend.
        vllm=dict(
            python_executable=sys.executable,
            source_dir=None,
            tensor_parallel_size=4,
            max_num_batched_tokens=8192,
            gpu_memory_utilization=0.8,
            load_format="instanttensor",
            timeout_seconds=86400,
            raw_cache_dir=None,
        ),
    ),
    num_train_epochs=1,
    # Derive the full schedule from the usable dataset by default. Launchers
    # may set a positive max_train_steps for bounded diagnostics.
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    parallel=dict(
        dp_replicate=runtime_node_count,
        dp_shard=runtime_local_world_size,
        cp=1,
        tp=1,
        ep=runtime_draft_ep,
        expert_tp=1,
        use_fsdp=True,
        context_parallel_backend="model_native",
        expert_dispatch_backend="native",
        reshard_after_forward=False,
        forward_prefetch=True,
        backward_prefetch=True,
        prefetch_depth=2,
        reduce_dtype="bf16",
        fsdp_wrap_granularity="block",
    ),
    target_parallel=dict(ep=runtime_draft_ep),
    # Both the reusable full-cache runner and bounded offline data batches use
    # a target mesh independent of draft training. TP remains fixed at four;
    # the DP-replicate/FSDP dimensions scale with the torchrun node layout.
    # EP overlays the target FSDP/TP rank domain so all 288 routed experts can
    # be loaded rank-locally while dense state remains TP4 + FSDP2 per node.
    offline_target_parallel=dict(
        dp_replicate=runtime_target_dp_replicate,
        dp_shard=runtime_target_dp_shard,
        cp=1,
        tp=4,
        ep=runtime_target_ep,
        expert_tp=1,
        use_fsdp=True,
    ),
    torch_compile=False,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=3000,
    save_checkpoints=True,
)

profiling = dict(enabled=False)

data = dict(
    online_target=False,
    # Keep production text-only by default; visual debug runs opt in through
    # MULTIMODAL=true and resolve relative media paths under media_root.
    multimodal=False,
    media_root=None,
    media_uri_map=None,
    # Preserve target-first/offline semantics without materializing the full
    # dataset: generate one bounded cache partition, train it, then delete it.
    offline_target_data_batches=True,
    train_data_path=(
        "train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl"
    ),
    source_jsonl_path=(
        "train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl"
    ),
    jsonl_index_cache_dir=None,
    data_batch_cache_dir=os.environ.get("DEEPSPEC_DATA_BATCH_CACHE_DIR"),
    target_cache_path=None,
    store_target_last_hidden_states=True,
    chat_template="glm5_next",
    max_length=131072,
    min_loss_tokens=14,
    num_workers=1,
    prefetch_factor=1,
)


def finalize_cfg(cfg):
    for runtime_key in (
        "runtime_world_size",
        "runtime_local_world_size",
        "runtime_node_count",
        "runtime_draft_ep",
        "runtime_target_is_node_local",
        "runtime_target_dp_replicate",
        "runtime_target_dp_shard",
        "runtime_target_ep",
    ):
        cfg.pop(runtime_key, None)
    logging_cfg = dict(cfg["logging"])
    output_root = os.environ.get("DEEPSPEC_OUTPUT_ROOT")
    checkpoint_root = (
        os.path.join(output_root, "checkpoints") if output_root else BASE_CKPT_DIR
    )
    tensorboard_root = (
        os.path.join(output_root, "tensorboard") if output_root else BASE_TB_DIR
    )
    logging_cfg["checkpoint_dir"] = os.path.join(
        checkpoint_root,
        str(cfg["project_name"]),
        str(cfg["exp_name"]),
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        tensorboard_root,
        str(cfg["project_name"]),
        str(cfg["exp_name"]),
    )
    cfg["logging"] = logging_cfg
    return cfg
