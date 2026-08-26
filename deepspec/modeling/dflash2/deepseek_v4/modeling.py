from __future__ import annotations

import torch

from deepspec.modeling.dflash2.common import CandidateSelector, GroupedDynamicCausalConv
from deepspec.modeling.dspark.deepseek_v4.modeling import (
    DeepseekV4DSparkDecoderLayer,
    DeepseekV4DSparkModel,
)


class DeepseekV4DFlash2DecoderLayer(DeepseekV4DSparkDecoderLayer):
    """DeepSeek-V4 hyper-connection block with DFlash2 dynamic convolutions."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.draft_block_size = int(config.verification_block_size)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size, config.conv_kernel_size, config.conv_group_size
        )

    def forward(self, **kwargs):
        hidden_states = kwargs.pop("hidden_states")
        dtype = hidden_states.dtype

        post, combine, collapsed = self.attn_hc(hidden_states)
        normalized = self.input_layernorm(collapsed)
        normalized, kernel = self.attention_conv.prepare(
            normalized, block_size=self.draft_block_size
        )
        attention_output = self.self_attn(hidden_states=normalized, **kwargs)
        attention_output = self.attention_conv.finish(
            attention_output, kernel, block_size=self.draft_block_size
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(-2) + torch.matmul(
            combine.to(dtype).transpose(-1, -2), hidden_states
        )

        post, combine, collapsed = self.ffn_hc(hidden_states)
        normalized = self.post_attention_layernorm(collapsed)
        normalized, kernel = self.mlp_conv.prepare(
            normalized, block_size=self.draft_block_size
        )
        mlp_output = self.mlp(normalized)
        mlp_output = self.mlp_conv.finish(
            mlp_output, kernel, block_size=self.draft_block_size
        )
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            combine.to(dtype).transpose(-1, -2), hidden_states
        )


class DeepseekV4DFlash2Model(DeepseekV4DSparkModel):
    decoder_layer_cls = DeepseekV4DFlash2DecoderLayer
    _no_split_modules = ["DeepseekV4DFlash2DecoderLayer"]
    checkpoint_excludes_embedding_head = True
    checkpoint_architecture_name = "DeepseekV4DFlash2Model"

    def __init__(self, config):
        super().__init__(config)
        for layer in self.layers:
            layer.attention_conv.reset_to_identity()
            layer.mlp_conv.reset_to_identity()
        self.candidate_selector = CandidateSelector(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=config.initializer_range,
        )

    def build_auxiliary_training_outputs(
        self, *, hidden_states, draft_logits, previous_token_ids, target_ids, eval_mask
    ):
        return self.candidate_selector.training_outputs(
            hidden=hidden_states,
            logits=draft_logits,
            predecessor_ids=previous_token_ids,
            target_ids=target_ids,
            eval_mask=eval_mask,
        )

    def select_draft_tokens(self, *, hidden_states, draft_logits, anchor_ids, temperature):
        return self.candidate_selector.select(
            hidden=hidden_states,
            logits=draft_logits,
            anchor_ids=anchor_ids,
            temperature=temperature,
        )

    def filter_checkpoint_state_dict(self, state_dict):
        """Exclude target-owned embedding/head tensors from draft checkpoints."""

        return {
            key: value
            for key, value in state_dict.items()
            if key not in {"embed_tokens.weight", "lm_head.weight"}
        }


__all__ = ["DeepseekV4DFlash2DecoderLayer", "DeepseekV4DFlash2Model"]
