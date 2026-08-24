import os

from deepspec.trainer import Qwen3_8DFlash2Trainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


project_name = "deepspec"
exp_name = "dflash2_block8_qwen3_8_27b"
seed = 42

model = dict(
    target_model_name_or_path=(
        "/mnt/afs-agentpro/share/models/Qwen/Qwen3.8-27B"
    ),
    # Official DFlash2 Qwen3.8 layout: one verified token plus seven masks.
    verification_block_size=8,
    num_draft_layers=5,
    target_layer_ids=[5, 19, 33, 47, 61],
    mask_token_id=248070,
    num_anchors=512,

    # Two-tap grouped dynamic convolution around every Attention/MLP.
    conv_kernel_size=2,
    conv_group_size=16,

    # Low-rank adjacent-candidate path selector.
    selector_rank=256,
    selector_top_k=16,

    # Base DFlash CE plus teacher-forced path-selection CE.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    selector_loss_alpha=1.0,
    # Keep selector pressure on late positions to counter suffix decay.
    selector_loss_decay_gamma=None,
)

train = dict(
    trainer_cls=Qwen3_8DFlash2Trainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    # The bundled repeat60 JSONL is consumed once.
    num_train_epochs=1,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    # Balanced 8-GPU default; the launcher overrides both from CP size.
    fsdp_size=4,
    fsdp_layerwise=False,
    context_parallel_size=2,
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
    # Declared here so strict --opts validation can safely override them.
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
    # DFlash2 uses CE plus path selection and never consumes target final logits.
    store_target_last_hidden_states=False,
    # Filled by the launcher so cache identity can be checked before training.
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
