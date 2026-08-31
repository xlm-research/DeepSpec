import os

from deepspec.trainer import Qwen3_8DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


project_name = "deepspec"
exp_name = "dspark_block7_qwen3_8_27b"
seed = 42

model = dict(
    target_model_name_or_path=(
        "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"
    ),
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 16, 31, 46, 61],
    # The tokenizer ends at 248076 while the embedding table has 248320 rows.
    mask_token_id=248077,
    num_anchors=512,

    markov_rank=256,
    markov_head_type="vanilla",
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3_8DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    # Match the canonical DSpark training protocol used by the other Qwen runs.
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    fsdp_size=None,
    fsdp_layerwise=False,
    context_parallel_size=1,
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
    save_checkpoints=True,
    checkpoint_dir=None,
    tensorboard_dir=None,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=1,
    prefetch_factor=1,
    # The bundled JSONL has literal <image> text but no media file paths.
    multimodal=False,
    # DSpark's L1 and confidence losses consume the target final hidden state.
    store_target_last_hidden_states=True,
    source_jsonl_path=None,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    if not logging_cfg.get("checkpoint_dir"):
        logging_cfg["checkpoint_dir"] = os.path.join(
            BASE_CKPT_DIR,
            str(cfg["project_name"]),
            str(cfg["exp_name"]),
        )
    if not logging_cfg.get("tensorboard_dir"):
        logging_cfg["tensorboard_dir"] = os.path.join(
            BASE_TB_DIR,
            str(cfg["project_name"]),
            str(cfg["exp_name"]),
        )
    cfg["logging"] = logging_cfg
    return cfg
