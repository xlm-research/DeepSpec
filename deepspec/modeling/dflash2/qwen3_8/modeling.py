from __future__ import annotations

from typing import Optional, Tuple

import torch
from transformers.cache_utils import Cache
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import FlashAttentionKwargs
from typing_extensions import Unpack

from deepspec.modeling.dflash2.common import (
    CandidateSelector,
    GroupedDynamicCausalConv,
)
from deepspec.modeling.dspark.qwen3.modeling import (
    Qwen3DSparkDecoderLayer,
    Qwen3DSparkModel,
)


class Qwen3_8DFlash2DecoderLayer(Qwen3DSparkDecoderLayer):
    """Qwen3 draft layer with DFlash2 convs around Attention and MLP."""

    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.draft_block_size = int(config.verification_block_size)
        self.attention_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            config.conv_kernel_size,
            config.conv_group_size,
        )
        self.mlp_conv = GroupedDynamicCausalConv(
            config.hidden_size,
            config.conv_kernel_size,
            config.conv_group_size,
        )

    def forward(
        self,
        target_hidden_states: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        context_position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,
        context_parallel_group=None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_kernel = self.attention_conv.prepare(
            hidden_states,
            block_size=self.draft_block_size,
        )
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            context_position_embeddings=context_position_embeddings,
            context_parallel_group=context_parallel_group,
            **kwargs,
        )[0]
        hidden_states = self.attention_conv.finish(
            hidden_states,
            attention_kernel,
            block_size=self.draft_block_size,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, mlp_kernel = self.mlp_conv.prepare(
            hidden_states,
            block_size=self.draft_block_size,
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(
            hidden_states,
            mlp_kernel,
            block_size=self.draft_block_size,
        )
        return residual + hidden_states


class Qwen3_8DFlash2Model(Qwen3DSparkModel):
    """Trainable DFlash2 drafter for the Qwen3.8-27B target."""

    config_class = Qwen3Config
    decoder_layer_cls = Qwen3_8DFlash2DecoderLayer
    _no_split_modules = ["Qwen3_8DFlash2DecoderLayer"]
    checkpoint_excludes_embedding_head = True
    checkpoint_architecture_name = "DFlash2DraftModel"

    def __init__(self, config) -> None:
        dflash_config = dict(getattr(config, "dflash_config", {}) or {})
        verification_block_size = int(
            getattr(
                config,
                "verification_block_size",
                dflash_config.get("block_size", 8),
            )
        )
        config.verification_block_size = verification_block_size
        config.proposal_hidden_offset = int(
            getattr(config, "proposal_hidden_offset", 1)
        )
        config.block_size = int(
            getattr(config, "block_size", verification_block_size - 1)
        )
        config.target_layer_ids = list(
            getattr(
                config,
                "target_layer_ids",
                dflash_config.get("target_layer_ids", [5, 19, 33, 47, 61]),
            )
        )
        config.mask_token_id = int(
            getattr(
                config,
                "mask_token_id",
                dflash_config.get("mask_token_id", 248070),
            )
        )
        config.num_anchors = int(getattr(config, "num_anchors", 512))
        config.enable_confidence_head = False
        config.markov_rank = 0
        config.conv_kernel_size = int(
            getattr(
                config,
                "conv_kernel_size",
                dflash_config.get("conv_kernel_size", 2),
            )
        )
        config.conv_group_size = int(
            getattr(
                config,
                "conv_group_size",
                dflash_config.get("conv_group_size", 16),
            )
        )
        config.selector_rank = int(
            getattr(
                config,
                "selector_rank",
                dflash_config.get("selector_rank", 256),
            )
        )
        config.selector_top_k = int(
            getattr(
                config,
                "selector_top_k",
                dflash_config.get("selector_top_k", 16),
            )
        )
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
        self,
        *,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return self.candidate_selector.training_outputs(
            hidden=hidden_states,
            logits=draft_logits,
            predecessor_ids=previous_token_ids,
            target_ids=target_ids,
            eval_mask=eval_mask,
        )

    def select_draft_tokens(
        self,
        *,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        anchor_ids: torch.Tensor,
        temperature: float,
    ):
        return self.candidate_selector.select(
            hidden=hidden_states,
            logits=draft_logits,
            anchor_ids=anchor_ids,
            temperature=temperature,
        )

    def filter_checkpoint_state_dict(self, state_dict):
        return {
            key: value
            for key, value in state_dict.items()
            if key not in {"embed_tokens.weight", "lm_head.weight"}
        }

    def save_pretrained(
        self, save_directory, *args, state_dict=None, **kwargs
    ):
        self.config.architectures = [self.checkpoint_architecture_name]
        if state_dict is not None:
            state_dict = self.filter_checkpoint_state_dict(state_dict)
        return super().save_pretrained(
            save_directory,
            *args,
            state_dict=state_dict,
            **kwargs,
        )


__all__ = [
    "Qwen3_8DFlash2DecoderLayer",
    "Qwen3_8DFlash2Model",
]
