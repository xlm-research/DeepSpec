import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


def build_draft_config(target_config, model_args):
    if str(target_config.model_type) != "deepseek_v4":
        raise ValueError(
            "DeepSeek-V4 DSpark expects model_type='deepseek_v4', got "
            f"{target_config.model_type!r}."
        )
    draft_config = copy.deepcopy(target_config)
    num_target_layers = int(target_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    if num_draft_layers < 1:
        raise ValueError("model.num_draft_layers must be positive.")

    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )
    confidence_head_alpha = float(model_args.confidence_head_alpha)
    if confidence_head_alpha < 0.0:
        raise ValueError("model.confidence_head_alpha cannot be negative.")
    markov_rank = int(model_args.markov_rank)
    if markov_rank < 0:
        raise ValueError("model.markov_rank cannot be negative.")

    draft_config.architectures = ["DeepseekV4DSparkModel"]
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # The DSpark paper uses a three-layer MoE backbone with a 128-token
    # sliding window.  CSA/HCA belong to the full target model, not the draft.
    draft_config.layer_types = ["sliding_attention"] * num_draft_layers
    draft_config.mlp_layer_types = ["moe"] * num_draft_layers
    # DeepseekV4DSparkModel is constructed directly instead of through
    # ``from_pretrained``.  Select the grouped expert dispatcher explicitly so
    # the draft does not inherit/fall back to the per-expert eager Python loop.
    draft_config._experts_implementation = "grouped_mm"
    draft_config.sliding_window = int(
        model_args.get("sliding_window", target_config.sliding_window)
    )
    draft_config.block_size = int(model_args.block_size)
    draft_config.tie_word_embeddings = False
    draft_config.use_cache = False
    draft_config._attn_implementation = "eager"
    # Construct a trainable BF16 model rather than inheriting the target's FP8
    # checkpoint quantizer.  Only embedding/head tensors are copied below.
    if hasattr(draft_config, "quantization_config"):
        draft_config.quantization_config = None
    draft_config.mask_token_id = int(model_args.mask_token_id)
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = int(model_args.num_anchors)
    draft_config.enable_confidence_head = confidence_head_alpha > 0.0
    if draft_config.enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = ["build_draft_config"]
