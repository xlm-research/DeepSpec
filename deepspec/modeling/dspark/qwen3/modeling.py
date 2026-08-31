from typing import Callable, Optional

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.attention.flex_attention import AuxRequest, flex_attention

from transformers.cache_utils import Cache
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    FlashAttentionKwargs,
    GradientCheckpointingLayer,
    Qwen3MLP,
    Qwen3PreTrainedModel,
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    eager_attention_forward,
    rotate_half,
)
from typing_extensions import Tuple, Unpack

from deepspec.modeling.dspark.common import (
    AcceptRatePredictor,
    DSparkContextParallelAttentionMask,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_dspark_context_parallel_attention_mask,
    create_noise_embed,
    create_position_ids,
    log_sampler_stats,
    sample_anchor_positions,
)
from deepspec.modeling.dspark.markov_head import build_markov_head
from deepspec.utils.sampling import sample_tokens


def _flex_attention_impl(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, aux = flex_attention(
        query,
        key,
        value,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_aux=AuxRequest(lse=True),
    )
    return output, aux.lse


def _context_flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flex_attention_impl(
        query,
        key,
        value,
        block_mask,
        scale,
        enable_gqa,
    )


def _draft_flex_attention_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flex_attention_impl(
        query,
        key,
        value,
        block_mask,
        scale,
        enable_gqa,
    )


def _context_flex_attention_recompute(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flex_attention_impl(
        query,
        key,
        value,
        block_mask,
        scale,
        enable_gqa,
    )


def _draft_flex_attention_recompute(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_mask,
    scale: float,
    enable_gqa: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _flex_attention_impl(
        query,
        key,
        value,
        block_mask,
        scale,
        enable_gqa,
    )


# DSpark CP uses FlexAttention's sparse kernel for each ring shard.  The
# surrounding model deliberately remains uncompiled in the distributed CP
# path, so compile the small kernel wrappers explicitly.  Keep context/draft
# and forward/backward-recompute wrappers on distinct Python code objects:
# Dynamo compile entries are code-object scoped, while their masks and
# requires_grad signatures are intentionally different.
_compiled_context_flex_attention_forward = torch.compile(
    _context_flex_attention_forward,
    dynamic=True,
)
_compiled_draft_flex_attention_forward = torch.compile(
    _draft_flex_attention_forward,
    dynamic=True,
)
_compiled_context_flex_attention_recompute = torch.compile(
    _context_flex_attention_recompute,
    dynamic=True,
)
_compiled_draft_flex_attention_recompute = torch.compile(
    _draft_flex_attention_recompute,
    dynamic=True,
)


def _ring_rotate_tensors(tensors: tuple[torch.Tensor, ...], group):
    """Rotate tensors from CP rank r to r+1 with one batched P2P call."""

    group_rank = dist.get_rank(group)
    group_size = dist.get_world_size(group)
    next_global_rank = dist.get_global_rank(group, (group_rank + 1) % group_size)
    previous_global_rank = dist.get_global_rank(
        group, (group_rank - 1) % group_size
    )
    received = tuple(
        torch.empty(
            tensor.shape,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        for tensor in tensors
    )
    operations = []
    for tensor, receive_buffer in zip(tensors, received):
        operations.append(
            dist.P2POp(
                dist.isend,
                tensor.contiguous(),
                next_global_rank,
                group=group,
            )
        )
        operations.append(
            dist.P2POp(
                dist.irecv,
                receive_buffer,
                previous_global_rank,
                group=group,
            )
        )
    for work in dist.batch_isend_irecv(operations):
        work.wait()
    return received


def _merge_attention_partials(
    partial_outputs: list[torch.Tensor],
    partial_lses: list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lse_stack = torch.stack(partial_lses, dim=0).to(torch.float32)
    global_lse = torch.logsumexp(lse_stack, dim=0)
    finite = torch.isfinite(global_lse).unsqueeze(0)
    weights = torch.where(
        finite,
        torch.exp(lse_stack - global_lse.unsqueeze(0)),
        torch.zeros_like(lse_stack),
    )
    output_float = torch.zeros_like(partial_outputs[0], dtype=torch.float32)
    for partial_output, weight in zip(partial_outputs, weights.unbind(0)):
        output_float.add_(partial_output.to(torch.float32) * weight.unsqueeze(-1))
    output = output_float.to(partial_outputs[0].dtype)
    return output, global_lse, lse_stack


class _RingFlexAttention(torch.autograd.Function):
    """Exact ring FlexAttention over sharded DSpark target context.

    Only one context K/V shard is resident per CP rank at a time.  Forward
    merges shard-local softmax outputs with their LSE values.  Backward
    recomputes one shard at a time and rotates the accumulated K/V gradients
    back to the rank that owns the shard.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        context_key: torch.Tensor,
        context_value: torch.Tensor,
        draft_key: torch.Tensor,
        draft_value: torch.Tensor,
        context_masks: tuple[object, ...],
        draft_mask,
        scale: float,
        enable_gqa: bool,
        group,
    ) -> torch.Tensor:
        group_rank = dist.get_rank(group)
        group_size = dist.get_world_size(group)
        if len(context_masks) != group_size:
            raise ValueError(
                "Expected one DSpark context mask per CP rank, got "
                f"{len(context_masks)} masks for CP size {group_size}."
            )

        partial_outputs = []
        partial_lses = []
        current_key = context_key
        current_value = context_value
        for step in range(group_size):
            source_rank = (group_rank - step) % group_size
            partial_output, partial_lse = (
                _compiled_context_flex_attention_forward(
                    query,
                    current_key,
                    current_value,
                    context_masks[source_rank],
                    scale,
                    enable_gqa,
                )
            )
            partial_outputs.append(partial_output)
            partial_lses.append(partial_lse)
            if step + 1 < group_size:
                current_key, current_value = _ring_rotate_tensors(
                    (current_key, current_value),
                    group,
                )

        draft_output, draft_lse = _compiled_draft_flex_attention_forward(
            query,
            draft_key,
            draft_value,
            draft_mask,
            scale,
            enable_gqa,
        )
        partial_outputs.append(draft_output)
        partial_lses.append(draft_lse)
        output, global_lse, lse_stack = _merge_attention_partials(
            partial_outputs,
            partial_lses,
        )

        ctx.save_for_backward(
            query,
            context_key,
            context_value,
            draft_key,
            draft_value,
            output,
            global_lse,
            lse_stack,
        )
        ctx.context_masks = context_masks
        ctx.draft_mask = draft_mask
        ctx.scale = float(scale)
        ctx.enable_gqa = bool(enable_gqa)
        ctx.group = group
        ctx.group_rank = group_rank
        ctx.group_size = group_size
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (
            query,
            context_key,
            context_value,
            draft_key,
            draft_value,
            output,
            global_lse,
            lse_stack,
        ) = ctx.saved_tensors
        group = ctx.group
        group_rank = ctx.group_rank
        group_size = ctx.group_size

        query_for_grad = query.detach().requires_grad_(True)
        grad_query = torch.zeros_like(query)
        current_key = context_key.detach()
        current_value = context_value.detach()
        current_grad_key = torch.zeros_like(context_key)
        current_grad_value = torch.zeros_like(context_value)

        def partial_gradients(
            key,
            value,
            block_mask,
            partial_lse,
            compiled_attention,
        ):
            key_for_grad = key.detach().requires_grad_(True)
            value_for_grad = value.detach().requires_grad_(True)
            with torch.enable_grad():
                partial_output, recomputed_lse = (
                    compiled_attention(
                        query_for_grad,
                        key_for_grad,
                        value_for_grad,
                        block_mask,
                        ctx.scale,
                        ctx.enable_gqa,
                    )
                )
            finite = torch.isfinite(global_lse)
            weight = torch.where(
                finite,
                torch.exp(partial_lse - global_lse),
                torch.zeros_like(global_lse),
            )
            grad_partial_output = grad_output * weight.unsqueeze(-1).to(
                grad_output.dtype
            )
            grad_partial_lse = weight * (
                grad_output.to(torch.float32)
                * (partial_output.to(torch.float32) - output.to(torch.float32))
            ).sum(dim=-1)
            return torch.autograd.grad(
                (partial_output, recomputed_lse),
                (query_for_grad, key_for_grad, value_for_grad),
                grad_outputs=(
                    grad_partial_output,
                    grad_partial_lse.to(recomputed_lse.dtype),
                ),
            )

        for step in range(group_size):
            source_rank = (group_rank - step) % group_size
            query_grad, key_grad, value_grad = partial_gradients(
                current_key,
                current_value,
                ctx.context_masks[source_rank],
                lse_stack[step],
                _compiled_context_flex_attention_recompute,
            )
            grad_query.add_(query_grad)
            current_grad_key.add_(key_grad)
            current_grad_value.add_(value_grad)
            # Rotate once after every contribution.  The final rotation sends
            # each accumulated gradient back to its original shard owner.
            (
                current_key,
                current_value,
                current_grad_key,
                current_grad_value,
            ) = _ring_rotate_tensors(
                (
                    current_key,
                    current_value,
                    current_grad_key,
                    current_grad_value,
                ),
                group,
            )

        query_grad, draft_key_grad, draft_value_grad = partial_gradients(
            draft_key,
            draft_value,
            ctx.draft_mask,
            lse_stack[group_size],
            _compiled_draft_flex_attention_recompute,
        )
        grad_query.add_(query_grad)
        return (
            grad_query,
            current_grad_key,
            current_grad_value,
            draft_key_grad,
            draft_value_grad,
            None,
            None,
            None,
            None,
            None,
        )


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DSparkAttention(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            self.num_attention_heads // self.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size,
            self.num_attention_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = (
            config.sliding_window
            if config.layer_types[layer_idx] == "sliding_attention"
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        context_position_embeddings: Optional[
            tuple[torch.Tensor, torch.Tensor]
        ] = None,
        context_parallel_group=None,
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_norm(
            self.k_proj(target_hidden_states).view(
                bsz, ctx_len, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        v_ctx = self.v_proj(target_hidden_states).view(
            bsz, ctx_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        k_noise = self.k_norm(
            self.k_proj(hidden_states).view(
                bsz, q_len, self.num_key_value_heads, self.head_dim
            )
        ).transpose(1, 2)
        v_noise = self.v_proj(hidden_states).view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        draft_cos, draft_sin = position_embeddings
        draft_cos = draft_cos[:, -q_len:]
        draft_sin = draft_sin[:, -q_len:]
        draft_cos = draft_cos.unsqueeze(1)
        draft_sin = draft_sin.unsqueeze(1)
        q = (q * draft_cos) + (rotate_half(q) * draft_sin)
        k_noise = (k_noise * draft_cos) + (rotate_half(k_noise) * draft_sin)

        if context_position_embeddings is None:
            # CP=1 compatibility path: context occupies the prefix of the
            # original combined position tensor.
            context_cos, context_sin = position_embeddings
            context_cos = context_cos[:, :ctx_len].unsqueeze(1)
            context_sin = context_sin[:, :ctx_len].unsqueeze(1)
        else:
            context_cos, context_sin = context_position_embeddings
            context_cos = context_cos.unsqueeze(1)
            context_sin = context_sin.unsqueeze(1)
        k_ctx = (k_ctx * context_cos) + (rotate_half(k_ctx) * context_sin)

        cp_enabled = context_parallel_group is not None and dist.get_world_size(
            context_parallel_group
        ) > 1
        if cp_enabled:
            if not isinstance(attention_mask, DSparkContextParallelAttentionMask):
                raise TypeError(
                    "DSpark ring CP requires DSparkContextParallelAttentionMask."
                )
            if past_key_values is not None:
                raise ValueError("DSpark ring CP training does not use a KV cache.")
            if self.config._attn_implementation != "flex_attention":
                raise ValueError(
                    "DSpark ring CP requires model attention implementation "
                    "'flex_attention'."
                )
            if self.training and self.attention_dropout != 0.0:
                raise ValueError(
                    "DSpark ring CP currently requires attention_dropout=0."
                )
            attn_output = _RingFlexAttention.apply(
                q,
                k_ctx,
                v_ctx,
                k_noise,
                v_noise,
                attention_mask.context_masks,
                attention_mask.draft_mask,
                self.scaling,
                self.num_key_value_groups > 1,
                context_parallel_group,
            )
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_weights = None
        else:
            k = torch.cat([k_ctx, k_noise], dim=-2)
            v = torch.cat([v_ctx, v_noise], dim=-2)
            if past_key_values is not None:
                cache_kwargs = {
                    "sin": draft_sin,
                    "cos": draft_cos,
                    "cache_position": cache_position,
                }
                k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
            if (
                self.config._attn_implementation == "flex_attention"
                and self.num_key_value_groups > 1
            ):
                kv_seq_len = k.shape[-2]
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)
                k = k.reshape(
                    bsz, self.num_attention_heads, kv_seq_len, self.head_dim
                )
                v = v.reshape(
                    bsz, self.num_attention_heads, kv_seq_len, self.head_dim
                )
            attn_fn: Callable = eager_attention_forward
            if self.config._attn_implementation != "eager":
                attn_fn = ALL_ATTENTION_FUNCTIONS[
                    self.config._attn_implementation
                ]
            attn_is_causal = bool(kwargs.get("is_causal", False))
            # The SDPA path may consult module.is_causal when dispatching
            # kernels, so mirror the per-call value on the module.
            self.is_causal = attn_is_causal
            kwargs["is_causal"] = attn_is_causal
            attn_output, attn_weights = attn_fn(
                self,
                q,
                k,
                v,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                sliding_window=self.sliding_window,
                **kwargs,
            )
        attn_output = attn_output.reshape(
            bsz, q_len, self.num_attention_heads * self.head_dim
        )
        return self.o_proj(attn_output), attn_weights


class Qwen3DSparkDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DSparkAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
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
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
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
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3DSparkModel(Qwen3PreTrainedModel):
    _no_split_modules = ["Qwen3DSparkDecoderLayer"]
    decoder_layer_cls = Qwen3DSparkDecoderLayer

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        required_fields = (
            "target_layer_ids",
            "mask_token_id",
            "num_anchors",
            "enable_confidence_head",
            "markov_rank",
        )
        for field in required_fields:
            assert hasattr(config, field), f"config.{field} must be provided."
        if int(config.markov_rank) > 0:
            assert hasattr(config, "markov_head_type"), (
                "config.markov_head_type must be provided when markov_rank > 0."
            )
        if bool(config.enable_confidence_head):
            assert hasattr(config, "confidence_head_with_markov"), (
                "config.confidence_head_with_markov must be provided when "
                "enable_confidence_head is true."
            )
        self.target_layer_ids = config.target_layer_ids
        self.context_parallel_size = 1
        self.context_parallel_rank = 0
        self.context_parallel_group = None
        self.model_parallel_group = None
        self.model_parallel_src_rank = 0

        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=getattr(config, "pad_token_id", None),
        )
        self.layers = nn.ModuleList(
            [
                self.decoder_layer_cls(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        layer_types = list(getattr(config, "layer_types", []))
        self.draft_sliding_window = (
            int(config.sliding_window)
            if layer_types
            and all(layer_type == "sliding_attention" for layer_type in layer_types)
            else None
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.block_size = int(config.block_size)
        self.verification_block_size = int(
            getattr(config, "verification_block_size", self.block_size)
        )
        self.proposal_hidden_offset = int(
            getattr(config, "proposal_hidden_offset", 0)
        )
        if self.proposal_hidden_offset < 0 or (
            self.proposal_hidden_offset + self.block_size
            > self.verification_block_size
        ):
            raise ValueError(
                "The proposal hidden-state slice must fit inside the draft "
                "verification block: "
                f"offset={self.proposal_hidden_offset}, "
                f"num_draft_tokens={self.block_size}, "
                f"verification_block_size={self.verification_block_size}."
            )
        self.mask_token_id = config.mask_token_id
        self.num_anchors = int(config.num_anchors)

        # Markov head.
        self.markov_head = build_markov_head(config)

        # Confidence head.
        self.enable_confidence_head = bool(config.enable_confidence_head)
        self.confidence_head_with_markov = False
        if self.enable_confidence_head:
            self.confidence_head_with_markov = bool(config.confidence_head_with_markov)
        if self.enable_confidence_head and self.confidence_head_with_markov:
            assert self.markov_head is not None

        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = int(config.hidden_size)
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)
        self.post_init()

    def configure_context_parallel(
        self,
        *,
        size: int,
        rank: int,
        group,
        model_parallel_group,
        model_parallel_src_rank: int,
    ):
        self.context_parallel_size = int(size)
        self.context_parallel_rank = int(rank)
        self.context_parallel_group = group
        self.model_parallel_group = model_parallel_group
        self.model_parallel_src_rank = int(model_parallel_src_rank)
        if self.num_anchors % self.context_parallel_size != 0:
            raise ValueError(
                "model.num_anchors must be divisible by train.context_parallel_size: "
                f"{self.num_anchors} % {self.context_parallel_size} != 0."
            )

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
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
    ):
        assert self.embed_tokens.weight.shape == embed_weight.shape
        assert self.lm_head.weight.shape == lm_head_weight.shape
        with torch.no_grad():
            self.embed_tokens.weight.copy_(embed_weight.detach())
            self.lm_head.weight.copy_(lm_head_weight.detach())
        if freeze:
            self.set_embedding_head_trainable(False)

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def build_auxiliary_training_outputs(
        self,
        *,
        hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        previous_token_ids: torch.Tensor,
        target_ids: torch.Tensor,
        eval_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del hidden_states, draft_logits, previous_token_ids, target_ids, eval_mask
        return {}

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                dtype=hidden_states.dtype
            )
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def sample_draft_tokens(
        self,
        base_logits: torch.Tensor,
        *,
        first_prev_token_ids: torch.Tensor,
        temperature: float = 0.0,
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, proposal_len = base_logits.shape[:2]
        if proposal_len == 0:
            empty_tokens = torch.empty(
                batch_size,
                0,
                dtype=torch.long,
                device=base_logits.device,
            )
            return empty_tokens, base_logits
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
        hidden_states: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert base_logits.ndim == 2, (
            "sample_draft_token_step expects base_logits shaped [batch, vocab], "
            f"got {tuple(base_logits.shape)}."
        )
        if self.markov_head is None:
            step_logits = base_logits
        else:
            step_logits = self.markov_head.apply_step_logits(
                base_logits,
                token_ids=prev_token_ids,
                hidden_states=hidden_states,
            )
        sampled_token_ids = sample_tokens(
            step_logits.unsqueeze(1),
            temperature=temperature,
        ).squeeze(1)
        return sampled_token_ids, step_logits

    def _forward_backbone(
        self,
        *,
        position_ids: torch.LongTensor,
        context_position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden_states: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.hidden_norm(self.fc(target_hidden_states))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        context_position_embeddings = None
        if context_position_ids is not None:
            context_position_embeddings = self.rotary_emb(
                target_hidden_states,
                context_position_ids,
            )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                context_position_embeddings=context_position_embeddings,
                context_parallel_group=(
                    self.context_parallel_group
                    if self.context_parallel_size > 1
                    else None
                ),
                **kwargs,
            )
        return self.norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: Optional[torch.Tensor] = None,
        context_chunk_len: Optional[torch.Tensor] = None,
        seq_len: Optional[torch.Tensor] = None,
    ) -> DSparkForwardOutput:
        bsz, padded_input_len = input_ids.shape
        device = input_ids.device

        if self.context_parallel_size > 1:
            if bsz != 1:
                raise ValueError(
                    "Context-parallel DSpark training currently requires "
                    "train.local_batch_size=1."
                )
            if any(
                value is None
                for value in (
                    context_chunk_len,
                    seq_len,
                )
            ):
                raise ValueError("Missing context-parallel cache metadata.")
            sequence_length = int(seq_len[0].item())
            local_context_len = int(context_chunk_len[0].item())
            if local_context_len % 2 != 0:
                raise ValueError(
                    "Native head/tail CP requires an even local context "
                    f"length, got {local_context_len}."
                )
            half_context_len = local_context_len // 2
            local_head_start = self.context_parallel_rank * half_context_len
            local_tail_start = (
                2 * self.context_parallel_size
                - self.context_parallel_rank
                - 1
            ) * half_context_len
        else:
            sequence_length = padded_input_len
            local_context_len = target_hidden_states.shape[1]

        if self.context_parallel_size == 1 or dist.get_rank() == self.model_parallel_src_rank:
            anchor_positions, block_keep_mask = sample_anchor_positions(
                seq_len=sequence_length,
                loss_mask=loss_mask[:, :sequence_length],
                num_anchors=self.num_anchors,
                device=device,
            )
        else:
            anchor_positions = torch.empty(
                (bsz, self.num_anchors), dtype=torch.long, device=device
            )
            block_keep_mask = torch.empty(
                (bsz, self.num_anchors), dtype=torch.bool, device=device
            )
        if self.context_parallel_size > 1:
            dist.broadcast(
                anchor_positions,
                src=self.model_parallel_src_rank,
                group=self.model_parallel_group,
            )
            dist.broadcast(
                block_keep_mask,
                src=self.model_parallel_src_rank,
                group=self.model_parallel_group,
            )
            anchors_per_rank = self.num_anchors // self.context_parallel_size
            anchor_start = self.context_parallel_rank * anchors_per_rank
            anchor_end = anchor_start + anchors_per_rank
            anchor_positions = anchor_positions[:, anchor_start:anchor_end]
            block_keep_mask = block_keep_mask[:, anchor_start:anchor_end]
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.verification_block_size,
        )
        draft_position_ids = create_position_ids(
            anchor_positions,
            self.verification_block_size,
        )
        if self.context_parallel_size > 1:
            context_position_ids = torch.cat(
                [
                    torch.arange(
                        local_head_start,
                        local_head_start + half_context_len,
                        device=device,
                    ),
                    torch.arange(
                        local_tail_start,
                        local_tail_start + half_context_len,
                        device=device,
                    ),
                ],
                dim=0,
            ).unsqueeze(0).expand(bsz, -1)
            backbone_position_ids = draft_position_ids
            dspark_attn_mask = create_dspark_context_parallel_attention_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                sequence_length=sequence_length,
                context_chunk_len=local_context_len,
                context_parallel_size=self.context_parallel_size,
                block_size=self.verification_block_size,
                device=device,
                sliding_window=self.draft_sliding_window,
            )
        else:
            context_position_ids = torch.arange(
                sequence_length, device=device
            ).unsqueeze(0).expand(bsz, -1)
            backbone_position_ids = torch.cat(
                [context_position_ids, draft_position_ids], dim=1
            )
            dspark_attn_mask = create_dspark_attention_mask(
                anchor_positions=anchor_positions,
                block_keep_mask=block_keep_mask,
                seq_len=sequence_length,
                block_size=self.verification_block_size,
                device=device,
                sliding_window=self.draft_sliding_window,
            )
        output_hidden = self._forward_backbone(
            position_ids=backbone_position_ids,
            context_position_ids=(
                context_position_ids if self.context_parallel_size > 1 else None
            ),
            noise_embedding=noise_embedding,
            target_hidden_states=target_hidden_states,
            attention_mask=dspark_attn_mask,
        )

        num_blocks = anchor_positions.size(1)
        verification_hidden_4d = output_hidden.reshape(
            bsz,
            num_blocks,
            self.verification_block_size,
            -1,
        )
        output_hidden_4d = verification_hidden_4d[
            :,
            :,
            self.proposal_hidden_offset : (
                self.proposal_hidden_offset + self.block_size
            ),
            :,
        ]

        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(
            1, 1, -1
        )
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets
        safe_label_indices = label_indices.clamp(max=sequence_length - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            if self.context_parallel_size > 1:
                # Every CP rank trains a different anchor subset.  Exchange
                # only those tiny index tensors, let each cache owner fill the
                # requested positions, then reduce the anchor-sized result.
                # No full-sequence hidden state is gathered.
                requested_indices = [
                    torch.empty_like(target_pred_indices)
                    for _ in range(self.context_parallel_size)
                ]
                dist.all_gather(
                    requested_indices,
                    target_pred_indices,
                    group=self.context_parallel_group,
                )
                all_target_pred_indices = torch.cat(requested_indices, dim=1)
                owns_head = (
                    (all_target_pred_indices >= local_head_start)
                    & (
                        all_target_pred_indices
                        < local_head_start + half_context_len
                    )
                )
                owns_tail = (
                    (all_target_pred_indices >= local_tail_start)
                    & (
                        all_target_pred_indices
                        < local_tail_start + half_context_len
                    )
                )
                local_pred_indices = torch.where(
                    owns_head,
                    all_target_pred_indices - local_head_start,
                    half_context_len
                    + all_target_pred_indices
                    - local_tail_start,
                ).clamp(min=0, max=local_context_len - 1)
                aligned_target_hidden = torch.gather(
                    target_last_hidden_states.unsqueeze(1).expand(
                        -1,
                        all_target_pred_indices.size(1),
                        -1,
                        -1,
                    ),
                    2,
                    local_pred_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        -1,
                        target_last_hidden_states.size(-1),
                    ),
                )
                owns_target = owns_head | owns_tail
                aligned_target_hidden = aligned_target_hidden * owns_target.unsqueeze(
                    -1
                )
                dist.all_reduce(
                    aligned_target_hidden,
                    op=dist.ReduceOp.SUM,
                    group=self.context_parallel_group,
                )
                aligned_target_hidden = aligned_target_hidden[
                    :, anchor_start:anchor_end
                ]
            else:
                aligned_target_hidden = torch.gather(
                    target_last_hidden_states.unsqueeze(1).expand(
                        -1,
                        anchor_positions.size(1),
                        -1,
                        -1,
                    ),
                    2,
                    target_pred_indices.unsqueeze(-1).expand(
                        -1,
                        -1,
                        -1,
                        target_last_hidden_states.size(-1),
                    ),
                )
            aligned_target_logits = self.compute_logits(aligned_target_hidden)
        eval_mask = build_eval_mask(
            seq_len=sequence_length,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )
        anchor_token_ids = torch.gather(
            input_ids,
            1,
            anchor_positions,
        )
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )
        draft_logits = self.compute_logits(output_hidden_4d)
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        log_sampler_stats(
            seq_len=sequence_length,
            loss_mask=loss_mask,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            block_size=self.block_size,
            num_anchors=anchor_positions.size(1),
        )

        confidence_pred = None
        if self.confidence_head is not None:
            if self.confidence_head_with_markov:
                prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(
                    dtype=output_hidden_4d.dtype
                )
                confidence_features = torch.cat(
                    [output_hidden_4d, prev_embeddings],
                    dim=-1,
                )
                confidence_pred = self.confidence_head(confidence_features).float()
            else:
                confidence_pred = self.confidence_head(output_hidden_4d).float()

        auxiliary_outputs = self.build_auxiliary_training_outputs(
            hidden_states=output_hidden_4d,
            draft_logits=draft_logits,
            previous_token_ids=prev_token_ids,
            target_ids=target_ids,
            eval_mask=eval_mask,
        )

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
            **auxiliary_outputs,
        )


__all__ = [
    "Qwen3DSparkModel",
    "Qwen3DSparkAttention",
    "Qwen3DSparkDecoderLayer",
]
