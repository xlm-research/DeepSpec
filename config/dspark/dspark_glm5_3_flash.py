import os
import math

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

model = dict(
    target_model_name_or_path=(
        "/mnt/afs-agentpro/share/models/zai-org/GLM-5.3-Flash"
    ),
    block_size=7,
    num_draft_layers=3,
    target_layer_ids=[0, 1, 2],
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
    # Requested optimizer-aligned target-cache partition count. The trainer
    # caps it at the remaining optimizer steps so short runs and resumes keep
    # every partition non-empty.
    data_batch_size=256,
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
    # target_layer_ids=[0, 1, 2] truncates GLM before its first MoE layer
    # (index 3), so the retained target has no experts to partition. Keep EP=1;
    # the draft's 288 routed experts use the independent EP=8 view above.
    target_parallel=dict(ep=1),
    # Both the reusable full-cache runner and bounded offline data batches use
    # a target mesh independent of draft training. TP remains fixed at four;
    # the DP-replicate/FSDP dimensions scale with the torchrun node layout.
    # EP remains 1 because the truncated target is dense.
    offline_target_parallel=dict(
        dp_replicate=runtime_target_dp_replicate,
        dp_shard=runtime_target_dp_shard,
        cp=1,
        tp=4,
        ep=1,
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
