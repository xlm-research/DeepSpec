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
            if config.loss.enable_vocab_parallel:
                self.hf_config["vocab_parallel"] = True
            else:
                self.hf_config.pop("vocab_parallel", None)
            if (
                config.loss.enable_vocab_parallel
                and config.parallelism.tensor_parallel_degree <= 1
            ):
                raise ValueError("Vocabulary parallel loss requires tensor parallelism")
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
        if parallel_dims.cp_enabled:
            from .features import context_positions

            length = tokens.shape[1]
            positions = context_positions(
                length, parallel_dims.cp, parallel_dims.get_mesh("cp").get_local_rank()
            ).to(tokens.device)
            for key in ("target_hidden_states", "target_last_hidden_states"):
                if key not in inputs or inputs[key] is None:
                    continue
                features = inputs[key]
                if features.shape[1] != length:
                    raise ValueError(
                        "CP preprocessing requires reconstructed full producer features"
                    )
                inputs[key] = (
                    features[:, positions.clamp_max(length - 1)]
                    * (positions < length)[None, :, None]
                )
            inputs["context_chunk_len"] = tokens.new_tensor([positions.numel()])
            inputs["seq_len"] = tokens.new_tensor([length])
            # Labels are unused by DSparkLoss; the native Trainer uses their
            # size to count consumed tokens. Count each original token once.
            labels = labels[:, positions[positions < length]]
        if parallel_dims.pp_enabled:
            if tokens.shape[1] != self.config.pipeline_sequence_length:
                raise ValueError(
                    "DSpark 1F1B requires fixed-length, padded feature batches"
                )
            if self.pipeline_stage == 1:
                inputs["input_ids"] = tokens
        return tokens, labels, inputs

    def forward(self, *args, **kwargs):
        if getattr(self, "pipeline_stage", None) != 1:
            return super().forward(*args, **kwargs)
        output = super().forward(
            kwargs.pop("input_ids"), **kwargs, pipeline_inputs=args
        )
        # PipelineStage sends and tracks tensor tuples. Supervision tensors are
        # returned alongside predictions so the loss retains its DSpark contract.
        empty = output.draft_logits.new_empty(0)
        return (
            output.draft_logits,
            output.target_ids,
            output.eval_mask,
            output.block_keep_mask,
            output.confidence_pred if output.confidence_pred is not None else empty,
            output.aligned_target_logits
            if output.aligned_target_logits is not None
            else empty,
        )


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
    from .pipeline import pipeline_draft

    return ModelSpec(
        name="dspark_draft",
        flavor="qwen3_8",
        model=build_draft_config(hf_config),
        max_context_length=hf_config["max_position_embeddings"],
        parallelize_fn=parallelize_draft,
        pipelining_fn=pipeline_draft,
        post_optimizer_build_fn=None,
        state_dict_adapter=DSparkStateDictAdapter,
    )
