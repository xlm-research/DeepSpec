from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from deepspec.modeling.dspark.qwen3.modeling import Qwen3DSparkModel


class Qwen3_6DSparkModel(Qwen3DSparkModel):
    """Full-attention DSpark draft paired with a Qwen3.6 hybrid target.

    The draft consumes cached hidden states from the target and does not need to
    duplicate the target's recurrent DeltaNet layer pattern. Reusing the stable
    Qwen3 DSpark block keeps the speculative algorithm independent from target
    internals while this class and its adapter own Qwen3.6-specific behavior.
    """

    config_class = Qwen3_5TextConfig


__all__ = ["Qwen3_6DSparkModel"]
