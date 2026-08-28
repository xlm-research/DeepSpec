import os

from deepspec.trainer import DeepseekV4DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec_128k"
exp_name = "dspark_deepseek_v4_flash_128k"
seed = 42

model = dict(
    target_model_name_or_path=(
        "/mnt/afs-agentpro/share/models/deepseek-ai/DeepSeek-V4-Flash-0731"
    ),
    block_size=7,
    num_draft_layers=3,
    target_layer_ids=[0, 1, 2],
    # V4 has no tokenizer mask token.  This checkpoint reserves embedding rows
    # above the tokenizer's 128000 entries; use the final reserved row.
    mask_token_id=129279,
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
    trainer_cls=DeepseekV4DSparkTrainer,
    # Keep the initially random routed draft stable across optimizer steps.
    lr=1.0e-5,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Pure EP overlays the base CP * TP * FSDP topology and does not multiply it.
    # Routed experts are excluded from TP/FSDP parameter shards.
    global_batch_size=8,
    # Split the planned training samples into this many near-equal disk-cache
    # blocks. Each block ends on a complete optimizer step.
    data_batch_size=3,
    # Each target-first data window still evaluates every sample exactly once.
    # Keep one epoch to avoid recomputing identical supervision across epochs.
    num_train_epochs=1,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    parallel=dict(
        dp_replicate=1,
        dp_shard=4,
        cp=2,
        tp=1,
        ep=1,
        expert_tp=1,
        use_fsdp=True,
        context_parallel_backend="model_native",
        expert_dispatch_backend="native",
        # The three 24.5-GB gathered draft blocks fit together on a 288-GB
        # B300. Keep them materialized across the gradient-accumulation window
        # and prefetch the static decoder order to remove exposed FSDP bubbles.
        reshard_after_forward=False,
        forward_prefetch=True,
        backward_prefetch=True,
        prefetch_depth=2,
        reduce_dtype="bf16",
        fsdp_wrap_granularity="block",
    ),
    target_parallel=dict(ep=8),
    # The distributed sliding-window path is shape-dynamic and performs ring
    # P2P collectives; whole-model compile is intentionally disabled.
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
    save_checkpoints=True,
)

profiling = dict(
    enabled=False,
    trace_dir="output/torch_profile",
    ranks=[0],
    # Schedule units are gradient-accumulation micro-steps.  The profile
    # launcher overrides these values from the runtime accumulation factor.
    skip_first_steps=0,
    wait_steps=0,
    warmup_steps=1,
    active_steps=1,
    repeat=1,
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
    with_flops=True,
    use_gzip=True,
    row_limit=100,
)

data = dict(
    online_target=True,
    train_data_path=("train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl"),
    jsonl_index_cache_dir=None,
    data_batch_cache_dir=None,
    target_cache_path=None,
    chat_template="deepseek_v4",
    max_length=131072,
    min_loss_tokens=14,
    num_workers=1,
    prefetch_factor=1,
)


def finalize_cfg(cfg):
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
