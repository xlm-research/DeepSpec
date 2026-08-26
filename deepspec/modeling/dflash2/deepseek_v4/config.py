from deepspec.modeling.dspark.deepseek_v4.config import build_draft_config as build_dspark_config


def build_draft_config(target_config, model_args):
    config = build_dspark_config(target_config, model_args)
    verification_block_size = int(
        model_args.get("verification_block_size", 8)
    )
    proposal_hidden_offset = int(model_args.get("proposal_hidden_offset", 1))
    proposal_block_size = int(
        model_args.get("block_size", verification_block_size - 1)
    )
    if verification_block_size < 2:
        raise ValueError("model.verification_block_size must be at least 2.")
    if proposal_hidden_offset != 1:
        raise ValueError(
            "DeepSeek-V4 DFlash2 reserves verification position 0 for the "
            "clean anchor, so model.proposal_hidden_offset must be 1."
        )
    if proposal_block_size != verification_block_size - 1:
        raise ValueError(
            "DeepSeek-V4 DFlash2 predicts every non-anchor verification "
            "position: model.block_size must equal "
            "model.verification_block_size - 1."
        )
    if int(config.num_hidden_layers) != len(config.target_layer_ids):
        raise ValueError(
            "DeepSeek-V4 DFlash2 requires one selected target feature per "
            "draft layer: "
            f"num_draft_layers={int(config.num_hidden_layers)}, "
            f"target_layer_ids={list(config.target_layer_ids)}."
        )

    conv_kernel_size = int(model_args.get("conv_kernel_size", 2))
    conv_group_size = int(model_args.get("conv_group_size", 16))
    selector_rank = int(model_args.get("selector_rank", 256))
    selector_top_k = int(model_args.get("selector_top_k", 16))
    if conv_kernel_size < 1:
        raise ValueError("model.conv_kernel_size must be positive.")
    if conv_group_size < 1 or int(config.hidden_size) % conv_group_size:
        raise ValueError(
            "model.conv_group_size must divide hidden_size: "
            f"{int(config.hidden_size)} % {conv_group_size} != 0."
        )
    if selector_rank < 1:
        raise ValueError("model.selector_rank must be positive.")
    if not 1 <= selector_top_k <= int(config.vocab_size):
        raise ValueError(
            "model.selector_top_k must be in [1, vocab_size]: "
            f"top_k={selector_top_k}, vocab_size={int(config.vocab_size)}."
        )

    config.architectures = ["DeepseekV4DFlash2Model"]
    config.verification_block_size = verification_block_size
    config.proposal_hidden_offset = proposal_hidden_offset
    config.block_size = proposal_block_size
    config.enable_confidence_head = False
    config.markov_rank = 0
    config.conv_kernel_size = conv_kernel_size
    config.conv_group_size = conv_group_size
    config.selector_rank = selector_rank
    config.selector_top_k = selector_top_k
    config.is_causal = False
    config.sample_from_anchor = False
    # Keep public DFlash2 metadata in the saved config. DeepSpec uses the
    # top-level mirrors above while serving/checkpoint tools can consume this
    # standard nested representation.
    config.dflash_config = {
        "block_size": verification_block_size,
        "conv_kernel_size": conv_kernel_size,
        "conv_group_size": conv_group_size,
        "mask_token_id": int(config.mask_token_id),
        "selector_rank": selector_rank,
        "selector_top_k": selector_top_k,
        "target_layer_ids": list(config.target_layer_ids),
    }
    return config


__all__ = ["build_draft_config"]
