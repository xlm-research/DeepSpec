import os

from deepspec.trainer import Qwen3_6DSparkTrainer
from deepspec.utils.constant import BASE_CKPT_DIR, BASE_TB_DIR


project_name = "deepspec"
exp_name = "dflash_block7_qwen3_6_27b"
seed = 42

model = dict(
    target_model_name_or_path="/mnt/afs-agentpro/share/models/Qwen/Qwen3.6-27B",
    block_size=7,
    num_draft_layers=5,
    target_layer_ids=[1, 16, 31, 46, 61],
    mask_token_id=248077,
    num_anchors=512,

    # DFlash does not use the DSpark Markov head.
    markov_rank=0,

    # DFlash does not use confidence-based early stopping.
    confidence_head_alpha=0.0,

    # DFlash trains the parallel block predictions with CE only.
    loss_decay_gamma=4.0,
    ce_loss_alpha=1.0,
    l1_loss_alpha=0.0,
)

train = dict(
    trainer_cls=Qwen3_6DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    # The repeat60 JSONL already supplies repeated examples, so consume the
    # generated offline cache once.
    num_train_epochs=1,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="full_shard",
    # Balanced 8-GPU default: CP=2 for sequence memory and FSDP=4 for params.
    # The one-click launcher derives FSDP from CONTEXT_PARALLEL_SIZE and
    # overrides both values together.
    fsdp_size=4,
    fsdp_layerwise=False,
    context_parallel_size=2,
    torch_compile=False,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=1,
    prefetch_factor=1,
    # The bundled JSONL has literal <image> text but no media file paths.
    multimodal=False,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg["project_name"])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(
        BASE_CKPT_DIR,
        project_name,
        exp_name,
    )
    logging_cfg["tensorboard_dir"] = os.path.join(
        BASE_TB_DIR,
        project_name,
        exp_name,
    )
    cfg["logging"] = logging_cfg
    return cfg
