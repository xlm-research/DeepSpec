from deepspec.modeling.dspark.qwen3.config import (
    build_draft_config as build_qwen3_draft_config,
)


def build_draft_config(target_config, model_args):
    """Build the target-conditioned Qwen3.6 DSpark draft configuration."""
    draft_config = build_qwen3_draft_config(
        target_config=target_config,
        model_args=model_args,
    )
    draft_config.architectures = ["Qwen3_6DSparkModel"]
    draft_config.deepspec_target_family = "qwen3_6"
    return draft_config


__all__ = ["build_draft_config"]
