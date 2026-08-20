import os

from deepspec.trainer import DeepseekV4DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec_128k"
exp_name = "dspark_deepseek_v4_flash_128k"
seed = 42

model = dict(
    target_model_name_or_path=(
        "/mnt/afs_agents/share_models/deepseek-ai/DeepSeek-V4-Flash"
    ),
    block_size=5,
    num_draft_layers=3,
    target_layer_ids=[40, 41, 42],
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
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    # Pure EP overlays the base CP * TP * FSDP topology and does not multiply it.
    # Routed experts are excluded from TP/FSDP parameter shards.
    global_batch_size=8,
    # Online distillation evaluates the frozen target once immediately before
    # each draft micro-batch.  Keep one epoch so every sample needs one target
    # forward rather than recomputing identical supervision across epochs.
    num_train_epochs=1,
    max_train_steps=1,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    fsdp_size=8,
    fsdp_layerwise=True,
    context_parallel_size=1,
    expert_parallel_size=8,
    tensor_parallel_size=1,
    pure_expert_parallel=True,
    # The distributed sliding-window path is shape-dynamic and performs ring
    # P2P collectives; whole-model compile is intentionally disabled.
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    online_target=True,
    train_data_path=None,
    target_cache_path=None,
    chat_template="deepseek_v4",
    max_length=131072,
    min_loss_tokens=14,
    # Group similarly sized records across synchronized sample ranks.  Raw
    # JSONL byte length is used as a zero-tokenization-cost sequence proxy.
    length_bucket_size=512,
    num_workers=1,
    prefetch_factor=1,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    output_root = os.environ.get("DEEPSPEC_OUTPUT_ROOT")
    checkpoint_root = (
        os.path.join(output_root, "checkpoints")
        if output_root
        else BASE_CKPT_DIR
    )
    tensorboard_root = (
        os.path.join(output_root, "tensorboard")
        if output_root
        else BASE_TB_DIR
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
