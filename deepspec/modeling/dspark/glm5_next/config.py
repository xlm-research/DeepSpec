import copy

from deepspec.modeling.dspark.common import validate_target_layer_ids


_EXPECTED_TARGET_FIELDS = {
    "hidden_size": 4096,
    "vocab_size": 154880,
    "num_hidden_layers": 45,
    "intermediate_size": 12288,
    "moe_intermediate_size": 2048,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "q_lora_rank": 1536,
    "kv_lora_rank": 512,
    "qk_nope_head_dim": 256,
    "qk_rope_head_dim": 0,
    "v_head_dim": 256,
    "max_position_embeddings": 1048576,
    "n_routed_experts": 288,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "routed_scaling_factor": 2.5,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "norm_topk_prob": True,
    "hc_mult": 4,
    "hc_eps": 1e-6,
    "hc_sinkhorn_iters": 20,
    "mhc": True,
    "mla_use_nope": True,
    "index_topk": 2048,
    "index_kpool": 4,
    "index_kpool_always_select_tail": True,
    "index_n_heads": 32,
    "index_head_dim": 128,
    "linear_num_heads": 64,
    "linear_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "linear_lower_bound": -5.0,
}


def validate_glm5_next_target_config(target_config):
    """Reject checkpoints that do not match the released GLM-5.3-Flash."""

    model_type = str(getattr(target_config, "model_type", ""))
    text_config = getattr(target_config, "text_config", None)
    text_model_type = str(getattr(text_config, "model_type", ""))
    mismatches = []
    if model_type != "glm5_next":
        mismatches.append(f"model_type={model_type!r} (expected 'glm5_next')")
    if list(getattr(target_config, "architectures", ()) or ()) != [
        "Glm5NextForConditionalGeneration"
    ]:
        mismatches.append(
            "architectures does not identify Glm5NextForConditionalGeneration"
        )
    if text_model_type != "glm5_next_text":
        mismatches.append(
            "text_config.model_type="
            f"{text_model_type!r} (expected 'glm5_next_text')"
        )
    if text_config is not None:
        for field, expected in _EXPECTED_TARGET_FIELDS.items():
            actual = getattr(text_config, field, None)
            if actual != expected:
                mismatches.append(
                    f"text_config.{field}={actual!r} (expected {expected!r})"
                )

        expected_attention = [
            "deepseek_sparse_attention"
            if (layer_idx + 1) % 4 == 0
            else "linear_attention"
            for layer_idx in range(45)
        ]
        if list(getattr(text_config, "layer_types", ()) or ()) != expected_attention:
            mismatches.append(
                "text_config.layer_types does not match GLM-5.3's "
                "3 KDA + 1 DSA schedule"
            )
        expected_mlp = ["dense"] * 3 + ["sparse"] * 42
        if list(getattr(text_config, "mlp_layer_types", ()) or ()) != expected_mlp:
            mismatches.append(
                "text_config.mlp_layer_types does not match 3 dense + 42 sparse layers"
            )

    if mismatches:
        raise ValueError(
            "GLM-5.3 DSpark expects the released GLM-5.3-Flash checkpoint; "
            "incompatible fields: "
            + "; ".join(mismatches)
            + "."
        )
    return text_config


def validate_glm5_next_mask_token(*, text_config, mask_token_id: int) -> int:
    """Validate the reserved embedding row used as DSpark's noise token."""

    mask_token_id = int(mask_token_id)
    expected = int(text_config.vocab_size) - 1
    if mask_token_id != expected:
        raise ValueError(
            "GLM-5.3 DSpark reserves the final vocabulary row as its mask "
            f"token: mask_token_id={mask_token_id}, expected {expected}."
        )
    return mask_token_id


def validate_glm5_next_tokenizer(tokenizer, *, mask_token_id: int) -> None:
    """Ensure the reserved DSpark row is not assigned to a tokenizer token."""

    mask_token_id = int(mask_token_id)
    used_ids = {int(token_id) for token_id in tokenizer.get_vocab().values()}
    if mask_token_id in used_ids:
        token = tokenizer.convert_ids_to_tokens(mask_token_id)
        raise ValueError(
            "GLM-5.3 DSpark mask_token_id conflicts with an effective "
            f"tokenizer token: id={mask_token_id}, token={token!r}."
        )
    if getattr(tokenizer, "mask_token_id", None) is not None:
        raise ValueError(
            "The released GLM-5.3 tokenizer is expected to have no configured "
            f"mask token, got mask_token_id={tokenizer.mask_token_id}."
        )


def build_draft_config(target_config, model_args):
    text_config = validate_glm5_next_target_config(target_config)

    draft_config = copy.deepcopy(text_config)
    num_target_layers = int(text_config.num_hidden_layers)
    num_draft_layers = int(model_args.num_draft_layers)
    if num_draft_layers < 1:
        raise ValueError("model.num_draft_layers must be positive.")

    target_layer_ids = validate_target_layer_ids(
        model_args.target_layer_ids,
        num_target_layers,
    )
    if num_target_layers - 1 in target_layer_ids:
        raise ValueError(
            "target_layer_ids must not include GLM's final decoder layer; its "
            "normalized final state is cached separately for L1/confidence loss."
        )
    confidence_head_alpha = float(model_args.confidence_head_alpha)
    if confidence_head_alpha < 0.0:
        raise ValueError("model.confidence_head_alpha cannot be negative.")
    markov_rank = int(model_args.markov_rank)
    if markov_rank < 0:
        raise ValueError("model.markov_rank cannot be negative.")

    block_size = int(model_args.block_size)
    if block_size != 7:
        raise ValueError(
            f"GLM-5.3 DSpark requires model.block_size=7, got {block_size}."
        )
    num_anchors = int(model_args.num_anchors)
    if num_anchors < 1:
        raise ValueError("model.num_anchors must be positive.")
    sliding_window = int(model_args.get("sliding_window", 128))
    if sliding_window < 1:
        raise ValueError("model.sliding_window must be positive.")
    mask_token_id = validate_glm5_next_mask_token(
        text_config=text_config,
        mask_token_id=model_args.mask_token_id,
    )

    draft_config.architectures = ["Glm5NextDSparkModel"]
    draft_config.deepspec_target_family = "glm5_3_flash"
    draft_config.deepspec_target_model_type = str(text_config.model_type)
    draft_config.deepspec_draft_architecture = "glm5_next_nope_window_attention"
    draft_config.target_context_layout = "contiguous"
    draft_config.deepspec_target_execution = "full_model"
    draft_config.deepspec_target_final_hidden_layer = num_target_layers - 1
    draft_config.num_target_layers = num_target_layers
    draft_config.num_hidden_layers = num_draft_layers
    # DSpark uses a short, sparse-MoE draft regardless of the target's hybrid
    # linear/DSA attention schedule. Its attention consumes frozen target
    # features through the bounded left-context path implemented below.
    draft_config.layer_types = ["deepseek_sparse_attention"] * num_draft_layers
    draft_config.mlp_layer_types = ["sparse"] * num_draft_layers
    draft_config._experts_implementation = "grouped_mm"
    draft_config.sliding_window = sliding_window
    draft_config.block_size = block_size
    draft_config.tie_word_embeddings = False
    draft_config.use_cache = False
    draft_config._attn_implementation = "eager"
    if hasattr(draft_config, "quantization_config"):
        draft_config.quantization_config = None
    draft_config.mask_token_id = mask_token_id
    draft_config.target_layer_ids = target_layer_ids
    draft_config.num_anchors = num_anchors
    draft_config.enable_confidence_head = confidence_head_alpha > 0.0
    if draft_config.enable_confidence_head:
        draft_config.confidence_head_with_markov = bool(
            model_args.confidence_head_with_markov
        )
    draft_config.markov_rank = markov_rank
    if markov_rank > 0:
        draft_config.markov_head_type = str(model_args.markov_head_type)
    return draft_config


__all__ = [
    "build_draft_config",
    "validate_glm5_next_mask_token",
    "validate_glm5_next_target_config",
    "validate_glm5_next_tokenizer",
]
