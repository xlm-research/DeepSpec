from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)


def build_draft_config(target_config, model_args):
    """Build a full-attention DSpark draft for the Qwen3.8-27B target."""

    target_text_config = getattr(target_config, "text_config", target_config)
    model_type = str(getattr(target_config, "model_type", ""))
    text_model_type = str(getattr(target_text_config, "model_type", ""))
    expected_fields = {
        "hidden_size": 5120,
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "intermediate_size": 17408,
        "num_attention_heads": 24,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "max_position_embeddings": 262144,
        "partial_rotary_factor": 0.25,
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
    }
    mismatches = []
    for field, expected in expected_fields.items():
        actual = getattr(target_text_config, field, None)
        if actual != expected:
            mismatches.append(f"{field}={actual!r} (expected {expected!r})")

    expected_layer_types = [
        "full_attention" if (layer_idx + 1) % 4 == 0 else "linear_attention"
        for layer_idx in range(64)
    ]
    layer_types = list(getattr(target_text_config, "layer_types", []) or [])
    if layer_types != expected_layer_types:
        mismatches.append(
            "layer_types does not match the 3 linear + 1 full-attention pattern"
        )

    rope_parameters = getattr(target_text_config, "rope_parameters", None)
    if rope_parameters is None:
        rope_parameters = getattr(target_text_config, "rope_scaling", None)
    rope_parameters = dict(rope_parameters or {})
    expected_rope_parameters = {
        "rope_type": "default",
        "rope_theta": 10_000_000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [11, 11, 10],
        "mrope_interleaved": True,
    }
    for field, expected in expected_rope_parameters.items():
        actual = rope_parameters.get(field)
        if actual != expected:
            mismatches.append(
                f"rope_parameters.{field}={actual!r} (expected {expected!r})"
            )

    valid_model_types = (
        (model_type == "qwen3_5" and text_model_type == "qwen3_5_text")
        or (model_type == "qwen3_5_text" and text_model_type == "qwen3_5_text")
    )
    if not valid_model_types:
        mismatches.append(
            f"model_type={model_type!r}, text_model_type={text_model_type!r}"
        )

    if mismatches:
        raise ValueError(
            "Qwen3.8-27B DSpark expects the released Qwen3.8-27B "
            "Qwen3.5-hybrid target config; incompatible fields: "
            + "; ".join(mismatches)
            + "."
        )

    draft_config = build_qwen3_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    # The target is a Qwen3.5 hybrid model, while the DSpark draft deliberately
    # uses the Qwen3 full-attention block and rotates the complete attention
    # head.  Keep the Qwen3.5 config class for existing checkpoint
    # compatibility, but do not serialize target-only MRoPE/partial-RoPE
    # metadata that the draft implementation does not consume.
    draft_config.rope_parameters = {
        "rope_type": "default",
        "rope_theta": float(expected_rope_parameters["rope_theta"]),
    }
    draft_config.partial_rotary_factor = 1.0
    draft_config.architectures = ["Qwen3_8DSparkModel"]
    draft_config.deepspec_target_family = "qwen3_8"
    draft_config.deepspec_draft_architecture = "qwen3_full_attention"
    draft_config.deepspec_draft_rope = "full_head"
    draft_config.deepspec_target_model_type = text_model_type
    draft_config.target_context_layout = "native_head_tail"
    return draft_config


__all__ = ["build_draft_config"]
