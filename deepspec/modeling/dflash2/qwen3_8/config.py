from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from deepspec.modeling.dspark.common import validate_target_layer_ids
from deepspec.modeling.dspark.qwen3.config import TRAIN_ATTN_IMPLEMENTATION


def build_draft_config(target_config, model_args):
    """Build the public Qwen3.8-27B DFlash2 draft architecture.

    Qwen3.8 uses a Qwen3.5 hybrid target, while its released DFlash2 drafter is
    a five-layer Qwen3-style sliding-attention model.  These dimensions follow
    ``doc/dflash2_qwen3.8.json`` rather than copying target attention fields.
    """

    target_text_config = getattr(target_config, "text_config", target_config)
    hidden_size = int(target_text_config.hidden_size)
    vocab_size = int(target_text_config.vocab_size)
    num_target_layers = int(target_text_config.num_hidden_layers)
    if (hidden_size, vocab_size, num_target_layers) != (5120, 248320, 64):
        raise ValueError(
            "Qwen3.8-27B DFlash2 expects target dimensions "
            "hidden_size=5120, vocab_size=248320, num_hidden_layers=64; got "
            f"{hidden_size}, {vocab_size}, {num_target_layers}."
        )

    verification_block_size = int(model_args.verification_block_size)
    if verification_block_size < 2:
        raise ValueError("model.verification_block_size must be at least 2.")
    num_draft_tokens = verification_block_size - 1
    num_draft_layers = int(model_args.num_draft_layers)
    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )
    if num_draft_layers != len(target_layer_ids):
        raise ValueError(
            "Qwen3.8 DFlash2 requires one selected target feature per draft "
            f"layer: {num_draft_layers} != {len(target_layer_ids)}."
        )

    rope_parameters = {
        "rope_theta": 10_000_000.0,
        "rope_type": "default",
    }
    dflash_config = {
        "block_size": verification_block_size,
        "conv_group_size": int(model_args.conv_group_size),
        "conv_kernel_size": int(model_args.conv_kernel_size),
        "mask_token_id": int(model_args.mask_token_id),
        "selector_rank": int(model_args.selector_rank),
        "selector_top_k": int(model_args.selector_top_k),
        "target_layer_ids": target_layer_ids,
    }
    draft_config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=17408,
        num_hidden_layers=num_draft_layers,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=262144,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        bos_token_id=None,
        eos_token_id=248044,
        pad_token_id=248044,
        rope_parameters=rope_parameters,
        sliding_window=2048,
        use_sliding_window=True,
        layer_types=["sliding_attention"] * num_draft_layers,
        is_causal=False,
        dtype="bfloat16",
    )
    draft_config.architectures = ["DFlash2DraftModel"]
    draft_config.dflash_config = dflash_config
    draft_config.num_target_layers = num_target_layers
    draft_config.max_window_layers = num_draft_layers

    # DeepSpec training-only fields.  The nested public block size is the
    # 8-token verification width; seven hidden positions become proposals.
    draft_config.block_size = num_draft_tokens
    draft_config.verification_block_size = verification_block_size
    draft_config.proposal_hidden_offset = 1
    draft_config.target_layer_ids = target_layer_ids
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.conv_group_size = int(model_args.conv_group_size)
    draft_config.conv_kernel_size = int(model_args.conv_kernel_size)
    draft_config.selector_rank = int(model_args.selector_rank)
    draft_config.selector_top_k = int(model_args.selector_top_k)
    draft_config.enable_confidence_head = False
    draft_config.markov_rank = 0
    draft_config._attn_implementation = TRAIN_ATTN_IMPLEMENTATION
    return draft_config


__all__ = ["build_draft_config"]
