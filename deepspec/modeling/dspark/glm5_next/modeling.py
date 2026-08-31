from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextPreTrainedModel,
    Glm5NextTextHyperConnection,
    Glm5NextTextHyperHead,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
    Glm5NextTextUnweightedRMSNorm,
)

from deepspec.modeling.dspark.deepseek_v4.modeling import (
    DeepseekV4DSparkModel,
    _gather_context_blocks,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.modeling.glm5_next_parallel import parallelize_glm5_next_model
from deepspec.modeling.dspark.common import AcceptRatePredictor


class Glm5NextDSparkAttention(nn.Module):
    """NoPE MQA attention over packed DSpark anchors and target context."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.qk_nope_head_dim)
        value_dim = int(config.v_head_dim)
        if value_dim != self.head_dim:
            raise ValueError(
                "GLM-5.3 DSpark requires v_head_dim == qk_nope_head_dim, got "
                f"{value_dim} != {self.head_dim}."
            )
        if int(config.qk_rope_head_dim) != 0:
            raise ValueError("GLM-5.3 DSpark expects the target's NoPE attention.")
        self.sliding_window = int(config.sliding_window)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)

        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False
        )
        self.q_a_norm = Glm5NextTextRMSNorm(
            config.q_lora_rank, eps=config.rms_norm_eps
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
        )
        self.q_b_norm = Glm5NextTextUnweightedRMSNorm(
            eps=config.rms_norm_eps
        )
        # DSpark uses one target-conditioned KV stream shared by all query
        # heads. This keeps its bounded context linear in sequence length.
        self.kv_proj = nn.Linear(
            config.hidden_size, self.head_dim, bias=False
        )
        self.kv_norm = Glm5NextTextRMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
        )
        self.sinks = nn.Parameter(torch.zeros(self.num_heads))

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_position_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch, query_length = hidden_states.shape[:2]
        num_blocks = int(anchor_positions.shape[1])
        if query_length != num_blocks * int(block_size):
            raise RuntimeError("Packed DSpark query length does not match its blocks.")

        q = self.q_b_proj(self.q_a_norm(self.q_a_proj(hidden_states))).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_b_norm(q)
        context_kv = self.kv_norm(self.kv_proj(target_hidden_states))
        noise_kv = self.kv_norm(self.kv_proj(hidden_states))

        offsets = torch.arange(self.sliding_window, device=hidden_states.device)
        wanted_positions = (
            anchor_positions[:, :, None] - self.sliding_window + offsets
        )
        context_start = context_position_ids[:, :1, None]
        context_indices = wanted_positions - context_start
        context_valid = (context_indices >= 0) & (
            context_indices < int(context_kv.shape[1])
        )
        context_indices = context_indices.clamp(
            min=0, max=max(int(context_kv.shape[1]) - 1, 0)
        )
        selected_context = _gather_context_blocks(context_kv, context_indices)
        noise_blocks = noise_kv.view(
            batch, num_blocks, int(block_size), self.head_dim
        )
        candidates = torch.cat([selected_context, noise_blocks], dim=2)
        valid = torch.cat(
            [
                context_valid,
                torch.ones(
                    (batch, num_blocks, int(block_size)),
                    dtype=torch.bool,
                    device=hidden_states.device,
                ),
            ],
            dim=2,
        )
        valid = valid & block_keep_mask.unsqueeze(-1)

        q_blocks = q.transpose(1, 2).reshape(
            batch, num_blocks, int(block_size), self.num_heads, self.head_dim
        )
        logits = torch.einsum("bnqhd,bnkd->bnhqk", q_blocks, candidates)
        logits = logits * float(self.scaling)
        logits = logits.masked_fill(~valid[:, :, None, None, :], float("-inf"))
        sinks = self.sinks.view(1, 1, self.num_heads, 1, 1).expand(
            batch, num_blocks, -1, int(block_size), -1
        )
        combined = torch.cat([logits, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probabilities = F.softmax(combined, dim=-1, dtype=torch.float32)[..., :-1]
        probabilities = F.dropout(
            probabilities,
            p=self.attention_dropout,
            training=self.training,
        ).to(candidates.dtype)
        output = torch.einsum("bnhqk,bnkd->bnqhd", probabilities, candidates)
        output = output.reshape(batch, query_length, -1)
        return self.o_proj(output)


class Glm5NextDSparkDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = Glm5NextDSparkAttention(config, layer_idx)
        self.mlp = Glm5NextTextMoE(config)
        self.input_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = Glm5NextTextHyperConnection(config)
        self.ffn_hc = Glm5NextTextHyperConnection(config)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        context_position_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        residual = hidden_states
        post, combine, collapsed = self.attn_hc(hidden_states)
        attention_output = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            target_hidden_states=target_hidden_states,
            context_position_ids=context_position_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=block_size,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(
            -2
        ) + torch.matmul(combine.to(dtype).transpose(-1, -2), residual)

        residual = hidden_states
        post, combine, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed))
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            combine.to(dtype).transpose(-1, -2), residual
        )


class Glm5NextDSparkModel(Glm5NextPreTrainedModel):
    _no_split_modules: ClassVar[list[str]] = ["Glm5NextDSparkDecoderLayer"]

    @classmethod
    def _can_set_experts_implementation(cls) -> bool:
        return True

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        if self.config._experts_implementation != "grouped_mm":
            raise RuntimeError(
                "GLM-5.3 draft MoE requires experts_implementation="
                f"'grouped_mm', got {self.config._experts_implementation!r}."
            )
        required = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required:
            if not hasattr(config, field):
                raise ValueError(f"config.{field} must be provided.")
        self.target_layer_ids = [int(value) for value in config.target_layer_ids]
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                Glm5NextDSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.hc_head = Glm5NextTextHyperHead()
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Glm5NextTextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.block_size = int(config.block_size)
        self.verification_block_size = self.block_size
        self.proposal_hidden_offset = 0
        self.mask_token_id = int(config.mask_token_id)
        self.num_anchors = int(config.num_anchors)
        self.local_num_anchors = self.num_anchors
        self.markov_head = build_markov_head(config)
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = bool(
            getattr(config, "confidence_head_with_markov", False)
        )
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                if self.markov_head is None:
                    raise ValueError(
                        "confidence_head_with_markov requires a Markov head."
                    )
                input_dim += int(config.markov_rank)
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

        self.context_parallel_size = 1
        self.context_parallel_rank = 0
        self.context_parallel_group = None
        self.expert_parallel_size = 1
        self.expert_parallel_rank = 0
        self.expert_parallel_group = None
        self.tensor_parallel_size = 1
        self.tensor_parallel_rank = 0
        self.tensor_parallel_group = None
        self.pure_expert_parallel = False
        self.post_init()

    # The sampling, loss-alignment, embedding initialization, and bounded CP
    # mechanics are model-independent DSpark behavior. Reuse the exercised V4
    # implementation while keeping GLM-specific modules above explicit.
    configure_context_parallel = DeepseekV4DSparkModel.configure_context_parallel
    _synchronize_anchor_sampling = DeepseekV4DSparkModel._synchronize_anchor_sampling
    initialize_embeddings_and_head = DeepseekV4DSparkModel.initialize_embeddings_and_head
    initialize_embedding_and_head_weights = (
        DeepseekV4DSparkModel.initialize_embedding_and_head_weights
    )
    set_embedding_head_trainable = DeepseekV4DSparkModel.set_embedding_head_trainable
    compute_logits = DeepseekV4DSparkModel.compute_logits
    build_auxiliary_training_outputs = (
        DeepseekV4DSparkModel.build_auxiliary_training_outputs
    )
    sample_draft_tokens = DeepseekV4DSparkModel.sample_draft_tokens
    sample_draft_token_step = DeepseekV4DSparkModel.sample_draft_token_step
    predict_confidence_step = DeepseekV4DSparkModel.predict_confidence_step
    _localize_context = DeepseekV4DSparkModel._localize_context
    forward = DeepseekV4DSparkModel.forward

    def configure_parallelism(self, topology) -> None:
        parallelize_glm5_next_model(self, topology=topology, draft=True)
        self.expert_parallel_size = int(topology.expert_parallel_size)
        self.expert_parallel_rank = int(topology.expert_parallel_rank)
        self.expert_parallel_group = topology.expert_parallel_group
        self.tensor_parallel_size = int(topology.tensor_parallel_size)
        self.tensor_parallel_rank = int(topology.tensor_parallel_rank)
        self.tensor_parallel_group = topology.tensor_parallel_group
        self.pure_expert_parallel = bool(topology.pure_expert_parallel)
        self.configure_context_parallel(
            size=topology.context_parallel_size,
            rank=topology.context_parallel_rank,
            group=topology.context_parallel_group,
            model_parallel_src_rank=topology.model_parallel_src_rank,
        )

    def apply_model_parallelism(self, *, context, config) -> None:
        sparse = context.sparse_mesh
        ep_group = None if sparse is None else sparse["ep"].get_group()
        ep_rank = 0 if sparse is None else sparse["ep"].get_local_rank()
        topology = SimpleNamespace(
            context_parallel_size=config.cp,
            context_parallel_rank=context.context_parallel_rank,
            context_parallel_group=context.cp_mesh.get_group(),
            model_parallel_src_rank=context.model_parallel_src_rank,
            expert_parallel_size=config.ep,
            expert_parallel_rank=ep_rank,
            expert_parallel_group=ep_group,
            tensor_parallel_size=config.tp,
            tensor_parallel_rank=context.tensor_parallel_rank,
            tensor_parallel_group=context.tp_mesh.get_group(),
            pure_expert_parallel=config.ep > 1,
        )
        self.configure_parallelism(topology)

    def _forward_backbone(
        self,
        *,
        noise_embedding: torch.Tensor,
        target_hidden_states: torch.Tensor,
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> torch.Tensor:
        del draft_position_ids
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        hidden_streams = noise_embedding.unsqueeze(2).expand(
            -1, -1, int(self.config.hc_mult), -1
        ).contiguous()
        for layer in self.layers:
            hidden_streams = layer(
                hidden_states=hidden_streams,
                target_hidden_states=target_hidden_states,
                context_position_ids=context_position_ids,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                block_size=self.verification_block_size,
            )
        return self.norm(self.hc_head(hidden_streams))


__all__ = [
    "Glm5NextDSparkAttention",
    "Glm5NextDSparkDecoderLayer",
    "Glm5NextDSparkModel",
]
