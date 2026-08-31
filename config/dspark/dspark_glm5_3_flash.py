import os

from deepspec.trainer import Glm5NextDSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR

project_name = "deepspec_glm5"
exp_name = "dspark_glm5_3_flash_128k"
seed = 42

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
    global_batch_size=8,
    # One target-first partition is convenient for the bundled 8-sample smoke
    # set. Production launchers should increase this to bound cache disk use.
    data_batch_size=1,
    num_train_epochs=1,
    max_train_steps=1,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    parallel=dict(
        dp_replicate=1,
        dp_shard=8,
        cp=1,
        tp=1,
        ep=8,
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
    # The retained GLM target layers are dense and FSDP-sharded. The draft's
    # 288 routed experts use the independent EP=8 sparse view above.
    target_parallel=dict(ep=1),
    torch_compile=False,
)

logging = dict(
    logging_steps=1,
    checkpointing_steps=3000,
    save_checkpoints=True,
)

profiling = dict(enabled=False)

data = dict(
    online_target=True,
    train_data_path="train_data/spec_o3_coldstartsft.first8.repeat1.deepspec.jsonl",
    jsonl_index_cache_dir=None,
    data_batch_cache_dir=None,
    target_cache_path=None,
    chat_template="glm5_next",
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
