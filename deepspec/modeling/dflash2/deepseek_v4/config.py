from deepspec.modeling.dspark.deepseek_v4.config import build_draft_config as build_dspark_config


def build_draft_config(target_config, model_args):
    config = build_dspark_config(target_config, model_args)
    config.architectures = ["DeepseekV4DFlash2Model"]
    config.verification_block_size = int(model_args.get("verification_block_size", 8))
    config.proposal_hidden_offset = int(model_args.get("proposal_hidden_offset", 1))
    config.block_size = int(model_args.get("block_size", config.verification_block_size - 1))
    config.enable_confidence_head = False
    config.markov_rank = 0
    config.conv_kernel_size = int(model_args.get("conv_kernel_size", 2))
    config.conv_group_size = int(model_args.get("conv_group_size", 16))
    config.selector_rank = int(model_args.get("selector_rank", 256))
    config.selector_top_k = int(model_args.get("selector_top_k", 16))
    return config


__all__ = ["build_draft_config"]
