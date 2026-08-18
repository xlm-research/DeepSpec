from __future__ import annotations

from typing import ClassVar

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
    DeepseekV4GroupedLinear,
    DeepseekV4HyperConnection,
    DeepseekV4HyperHead,
    DeepseekV4PreTrainedModel,
    DeepseekV4RMSNorm,
    DeepseekV4RotaryEmbedding,
    DeepseekV4SparseMoeBlock,
    DeepseekV4UnweightedRMSNorm,
)

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkForwardOutput,
    build_eval_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.modeling.deepseek_v4_parallel import (
    parallelize_deepseek_v4_model,
)
from deepspec.modeling.target.deepseek_v4_cp import ring_left_context
from deepspec.utils.sampling import sample_tokens


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_rotary(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = int(cos.shape[-1])
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (_rotate_half(rope).float() * sin)).to(
        x.dtype
    )
    return torch.cat([nope, rotated], dim=-1)


def _gather_context_blocks(
    context: torch.Tensor, indices: torch.Tensor
) -> torch.Tensor:
    _, blocks, _ = indices.shape
    expanded = context[:, None].expand(-1, blocks, -1, -1)
    return torch.gather(
        expanded,
        2,
        indices.unsqueeze(-1).expand(-1, -1, -1, context.shape[-1]),
    )


class DeepseekV4DSparkAttention(nn.Module):
    """V4 MQA attention over packed DSpark anchors and a 128-token context."""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = int(layer_idx)
        self.num_heads = int(config.num_attention_heads)
        self.head_dim = int(config.head_dim)
        self.sliding_window = int(config.sliding_window)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = float(config.attention_dropout)
        self._deepspec_local_o_groups = int(config.o_groups)

        self.q_a_proj = nn.Linear(
            config.hidden_size, config.q_lora_rank, bias=False
        )
        self.q_a_norm = DeepseekV4RMSNorm(
            config.q_lora_rank, eps=config.rms_norm_eps
        )
        self.q_b_proj = nn.Linear(
            config.q_lora_rank, self.num_heads * self.head_dim, bias=False
        )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.rms_norm_eps)
        self.kv_proj = nn.Linear(config.hidden_size, self.head_dim, bias=False)
        self.kv_norm = DeepseekV4RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )
        self.o_a_proj = DeepseekV4GroupedLinear(
            self.num_heads * self.head_dim // config.o_groups,
            config.o_groups * config.o_lora_rank,
            config.o_groups,
        )
        self.o_b_proj = nn.Linear(
            config.o_groups * config.o_lora_rank,
            config.hidden_size,
            bias=False,
        )
        self.sinks = nn.Parameter(torch.empty(self.num_heads))
        nn.init.zeros_(self.sinks)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        draft_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        context_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        batch, query_length = hidden_states.shape[:2]
        num_blocks = int(anchor_positions.shape[1])
        if query_length != num_blocks * int(block_size):
            raise RuntimeError("Packed DSpark query length does not match its blocks.")

        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_residual).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        q = self.q_b_norm(q)
        draft_cos, draft_sin = draft_position_embeddings
        q = _apply_rotary(q, draft_cos, draft_sin)

        context_kv = self.kv_norm(self.kv_proj(target_hidden_states)).unsqueeze(1)
        context_cos, context_sin = context_position_embeddings
        context_kv = _apply_rotary(context_kv, context_cos, context_sin)[:, 0]
        noise_kv = self.kv_norm(self.kv_proj(hidden_states)).unsqueeze(1)
        noise_kv = _apply_rotary(noise_kv, draft_cos, draft_sin)[:, 0]

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
        logits = logits.masked_fill(
            ~valid[:, :, None, None, :], float("-inf")
        )
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
        output = output.reshape(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        output = _apply_rotary(output, draft_cos, -draft_sin).transpose(1, 2)
        grouped = output.reshape(
            batch, query_length, self._deepspec_local_o_groups, -1
        )
        grouped = self.o_a_proj(grouped).flatten(2)
        return self.o_b_proj(grouped)


class DeepseekV4DSparkDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.layer_idx = int(layer_idx)
        self.self_attn = DeepseekV4DSparkAttention(config, layer_idx)
        self.mlp = DeepseekV4SparseMoeBlock(config, layer_idx)
        self.input_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attn_hc = DeepseekV4HyperConnection(config)
        self.ffn_hc = DeepseekV4HyperConnection(config)

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        draft_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        context_position_embeddings: tuple[torch.Tensor, torch.Tensor],
        draft_position_ids: torch.Tensor,
        context_position_ids: torch.Tensor,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
        block_size: int,
    ) -> torch.Tensor:
        dtype = hidden_states.dtype
        post, combine, collapsed = self.attn_hc(hidden_states)
        attention_output = self.self_attn(
            hidden_states=self.input_layernorm(collapsed),
            target_hidden_states=target_hidden_states,
            draft_position_embeddings=draft_position_embeddings,
            context_position_embeddings=context_position_embeddings,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            block_size=block_size,
        )
        hidden_states = post.to(dtype).unsqueeze(-1) * attention_output.unsqueeze(
            -2
        ) + torch.matmul(combine.to(dtype).transpose(-1, -2), hidden_states)

        post, combine, collapsed = self.ffn_hc(hidden_states)
        mlp_output = self.mlp(self.post_attention_layernorm(collapsed))
        return post.to(dtype).unsqueeze(-1) * mlp_output.unsqueeze(-2) + torch.matmul(
            combine.to(dtype).transpose(-1, -2), hidden_states
        )


class DeepseekV4DSparkModel(DeepseekV4PreTrainedModel):
    _no_split_modules: ClassVar[list[str]] = ["DeepseekV4DSparkDecoderLayer"]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, DeepseekV4DSparkAttention):
            nn.init.zeros_(module.sinks)

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
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
                DeepseekV4DSparkDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.rotary_emb = DeepseekV4RotaryEmbedding(config)
        self.hc_head = DeepseekV4HyperHead(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = DeepseekV4RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.block_size = int(config.block_size)
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
        self.post_init()

    def configure_context_parallel(
        self,
        *,
        size: int,
        rank: int,
        group,
        model_parallel_group=None,
        model_parallel_src_rank=None,
    ) -> None:
        del model_parallel_src_rank
        self.context_parallel_size = int(size)
        self.context_parallel_rank = int(rank)
        del model_parallel_group
        self.context_parallel_group = group
        base, remainder = divmod(self.num_anchors, self.context_parallel_size)
        self.local_num_anchors = base + int(self.context_parallel_rank < remainder)
        if self.local_num_anchors < 1:
            raise ValueError(
                "model.num_anchors must be at least train.context_parallel_size."
            )

    def configure_parallelism(self, topology) -> None:
        """Install orthogonal EP/TP shards, then configure the existing CP path."""

        parallelize_deepseek_v4_model(self, topology=topology, draft=True)
        self.expert_parallel_size = int(topology.expert_parallel_size)
        self.expert_parallel_rank = int(topology.expert_parallel_rank)
        self.expert_parallel_group = topology.expert_parallel_group
        self.tensor_parallel_size = int(topology.tensor_parallel_size)
        self.tensor_parallel_rank = int(topology.tensor_parallel_rank)
        self.tensor_parallel_group = topology.tensor_parallel_group
        self.configure_context_parallel(
            size=topology.context_parallel_size,
            rank=topology.context_parallel_rank,
            group=topology.context_parallel_group,
            model_parallel_src_rank=topology.model_parallel_src_rank,
        )

    def _synchronize_anchor_sampling(
        self,
        anchor_positions: torch.Tensor,
        block_keep_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Keep stochastic anchor choices identical across EP/TP peers."""

        for size, group in (
            (self.tensor_parallel_size, self.tensor_parallel_group),
            (self.expert_parallel_size, self.expert_parallel_group),
        ):
            if int(size) == 1:
                continue
            source_rank = (
                int(dist.get_global_rank(group, 0))
                if hasattr(dist, "get_global_rank")
                else int(dist.get_process_group_ranks(group)[0])
            )
            dist.broadcast(anchor_positions, src=source_rank, group=group)
            dist.broadcast(block_keep_mask, src=source_rank, group=group)
        return anchor_positions, block_keep_mask

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ) -> None:
        self.initialize_embedding_and_head_weights(
            embed_weight=embed_tokens.weight,
            lm_head_weight=lm_head.weight,
            freeze=freeze,
        )

    def initialize_embedding_and_head_weights(
        self,
        *,
        embed_weight: torch.Tensor,
        lm_head_weight: torch.Tensor,
        freeze: bool = True,
    ) -> None:
        if self.embed_tokens.weight.shape != embed_weight.shape:
            raise ValueError("Target and draft embedding shapes differ.")
        if self.lm_head.weight.shape != lm_head_weight.shape:
            raise ValueError("Target and draft LM-head shapes differ.")
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.detach())
            self.lm_head.weight.copy_(lm_head_weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool) -> None:
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if base_logits.shape[1] == 0:
            return (
                torch.empty(
                    base_logits.shape[0],
                    0,
                    dtype=torch.long,
                    device=base_logits.device,
                ),
                base_logits,
            )
        if self.markov_head is None:
            return sample_tokens(base_logits, temperature), base_logits
        return self.markov_head.sample_block_tokens(
            base_logits,
            first_prev_token_ids=first_prev_token_ids,
            hidden_states=hidden_states,
            temperature=temperature,
        )

    def sample_draft_token_step(
        self,
        base_logits: torch.Tensor,
        *,
        prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled = sample_tokens(
            step_logits.unsqueeze(1), temperature=temperature
        ).squeeze(1)
        return sampled, step_logits

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            if prev_token_ids is None:
                raise ValueError("prev_token_ids are required by the Markov head.")
            previous = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                hidden_states.dtype
            )
            hidden_states = torch.cat([hidden_states, previous], dim=-1)
        return self.confidence_head(hidden_states).float()

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
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        hidden_streams = noise_embedding.unsqueeze(2).expand(
            -1, -1, int(self.config.hc_mult), -1
        ).contiguous()
        draft_position_embeddings = self.rotary_emb(
            noise_embedding,
            position_ids=draft_position_ids,
            layer_type="main",
        )
        context_position_embeddings = self.rotary_emb(
            target_hidden_states,
            position_ids=context_position_ids,
            layer_type="main",
        )
        for layer in self.layers:
            hidden_streams = layer(
                hidden_states=hidden_streams,
                target_hidden_states=target_hidden_states,
                draft_position_embeddings=draft_position_embeddings,
                context_position_embeddings=context_position_embeddings,
                draft_position_ids=draft_position_ids,
                context_position_ids=context_position_ids,
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                block_size=self.block_size,
            )
        return self.norm(self.hc_head(hidden_streams))

    def _localize_context(
        self,
        *,
        input_ids: torch.Tensor,
        loss_mask: torch.Tensor,
        target_hidden_states: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None,
        context_start,
        context_len,
        seq_len,
    ):
        if self.context_parallel_size == 1:
            local_length = int(target_hidden_states.shape[1])
            return (
                input_ids[:, :local_length],
                loss_mask[:, :local_length],
                target_hidden_states[:, :local_length],
                None
                if target_last_hidden_states is None
                else target_last_hidden_states[:, :local_length],
                0,
                0,
            )
        if input_ids.shape[0] != 1:
            raise ValueError("DeepSeek-V4 draft CP currently requires batch size 1.")
        start = int(context_start.reshape(-1)[0].item())
        local_length = int(context_len.reshape(-1)[0].item())
        total_length = int(seq_len.reshape(-1)[0].item())
        local_target = target_hidden_states[:, :local_length]
        halo, halo_start = ring_left_context(
            local_target,
            local_start=start,
            sequence_length=total_length,
            window=int(self.config.sliding_window),
            group=self.context_parallel_group,
            rank=self.context_parallel_rank,
            size=self.context_parallel_size,
        )
        context = torch.cat([halo, local_target], dim=1)
        local_last = None
        if target_last_hidden_states is not None:
            local_last = target_last_hidden_states[:, :local_length]
        return (
            input_ids[:, start : start + local_length],
            loss_mask[:, start : start + local_length],
            context,
            local_last,
            start,
            halo_start,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None = None,
        context_start=None,
        context_len=None,
        seq_len=None,
    ) -> DSparkForwardOutput:
        (
            local_input_ids,
            local_loss_mask,
            context_hidden,
            local_last_hidden,
            local_start,
            context_global_start,
        ) = self._localize_context(
            input_ids=input_ids,
            loss_mask=loss_mask,
            target_hidden_states=target_hidden_states,
            target_last_hidden_states=target_last_hidden_states,
            context_start=context_start,
            context_len=context_len,
            seq_len=seq_len,
        )
        batch, local_length = local_input_ids.shape
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=local_length,
            loss_mask=local_loss_mask,
            num_anchors=self.local_num_anchors,
            device=input_ids.device,
        )
        anchor_positions, block_keep_mask = self._synchronize_anchor_sampling(
            anchor_positions, block_keep_mask
        )
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            local_input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )
        global_anchor_positions = anchor_positions + int(local_start)
        draft_position_ids = create_position_ids(
            global_anchor_positions, self.block_size
        )
        context_position_ids = torch.arange(
            context_global_start,
            context_global_start + int(context_hidden.shape[1]),
            device=input_ids.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(batch, -1)
        output_hidden = self._forward_backbone(
            noise_embedding=noise_embedding,
            target_hidden_states=context_hidden,
            draft_position_ids=draft_position_ids,
            context_position_ids=context_position_ids,
            anchor_positions=global_anchor_positions,
            block_keep_mask=block_keep_mask,
        )

        num_blocks = int(anchor_positions.shape[1])
        output_hidden_4d = output_hidden.view(
            batch, num_blocks, self.block_size, -1
        )
        label_offsets = torch.arange(
            1, self.block_size + 1, device=input_ids.device
        ).view(1, 1, -1)
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=local_length - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            local_input_ids.unsqueeze(1).expand(-1, num_blocks, -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if local_last_hidden is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                local_last_hidden.unsqueeze(1).expand(-1, num_blocks, -1, -1),
                2,
                target_pred_indices.unsqueeze(-1).expand(
                    -1, -1, -1, local_last_hidden.shape[-1]
                ),
            )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=local_length,
            loss_mask=local_loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            local_input_ids, 1, anchor_positions
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]], dim=-1
        )
        draft_logits = self.compute_logits(output_hidden).view(
            batch, num_blocks, self.block_size, -1
        )
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=local_length,
            loss_mask=local_loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=self.local_num_anchors,
        )
        confidence_pred = None
        if self.confidence_head is not None:
            features = output_hidden_4d
            if self.confidence_head_with_markov:
                previous = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    features.dtype
                )
                features = torch.cat([features, previous], dim=-1)
            confidence_pred = self.confidence_head(features).float()
        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


__all__ = [
    "DeepseekV4DSparkAttention",
    "DeepseekV4DSparkDecoderLayer",
    "DeepseekV4DSparkModel",
]
