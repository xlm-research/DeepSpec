"""DSpark draft model and native TorchTitan configuration adapter."""

from dataclasses import dataclass, field

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_spec import ModelSpec
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Qwen3DSparkDecoderLayer, Qwen3DSparkModel


class DSparkDraftModel(Qwen3DSparkModel):
    """Own the existing DSpark kernels while implementing Titan's model protocol."""

    config_class = Qwen3_5TextConfig

    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        hf_config: dict = field(default_factory=dict)

        def build(self, **kwargs):
            config = Qwen3_5TextConfig.from_dict(self.hf_config)
            config._attn_implementation = "flex_attention"
            # The retained DSpark initialization casts nonpersistent RoPE
            # buffers together with parameters. Preserve that rounding too.
            return DSparkDraftModel(config).to(dtype=torch.get_default_dtype())

        def update_from_config(self, *, config, **kwargs):
            if (
                config.training.max_context_length
                > self.hf_config["max_position_embeddings"]
            ):
                raise ValueError(
                    "Training context exceeds the draft model context limit"
                )

        def get_nparams_and_flops(self, model, seq_len):
            count = sum(parameter.numel() for parameter in model.parameters())
            # The dense estimate is for Titan's diagnostic MFU, not benchmarking.
            return count, 6 * count

    def __init__(self, config):
        super().__init__(config)
        self.set_embedding_head_trainable(False)

    def verify_module_protocol(self):
        # This adapter explicitly owns initialization of its HF-compatible
        # nn.Modules; they do not use Module.init_states recursion.
        if not all(isinstance(layer, Qwen3DSparkDecoderLayer) for layer in self.layers):
            raise TypeError("DSpark requires its dual-input decoder blocks")

    def init_weights(self, **kwargs):
        # Titan materializes a meta model with to_empty. HF initialization flags
        # belong to the old meta tensors and must not skip the materialized data.
        for tensor in (*self.parameters(), *self.buffers()):
            tensor._is_hf_initialized = False
        self.apply(self._init_weights)

    def preprocess_inputs(self, input_dict, *, parallel_dims, parallelism):
        inputs = dict(input_dict)
        labels = inputs.pop("labels")
        tokens = inputs.pop("input_ids")
        return tokens, labels, inputs


class DSparkStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config, hf_assets_path):
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        # Tokenizer assets belong to the teacher; its shard map is not the draft's.
        self.fqn_to_index_mapping = None

    def from_hf(self, state_dict):
        return dict(state_dict)

    def to_hf(self, state_dict):
        return dict(state_dict)


def build_draft_config(hf_config):
    return DSparkDraftModel.Config(hf_config=dict(hf_config))


def model_spec(hf_config):
    from .parallelize import parallelize_draft

    return ModelSpec(
        name="dspark_draft",
        flavor="qwen3_8",
        model=build_draft_config(hf_config),
        max_context_length=hf_config["max_position_embeddings"],
        parallelize_fn=parallelize_draft,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=DSparkStateDictAdapter,
    )
