"""Memory-bounded Ring Context Parallel for the DeepSeek-V4 target model.

DeepSeek-V4 does not use ordinary full attention.  Every layer combines a
128-token causal sliding window with either CSA or HCA compressed history.
This adapter therefore keeps the uncompressed sequence sharded, exchanges
only the small boundary halo, and rotates compressed shards around a CP ring.
CSA index scores are evaluated in bounded query/key tiles before the selected
compressed K/V entries are consumed.  No rank materializes the full 128K
uncompressed K/V sequence or gathers hidden states onto a leader.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn.functional as F

from deepspec.modeling.target.common import TargetForwardResult
from deepspec.utils.parallel import compute_context_parallel_range


def _group_global_rank(group, group_rank: int) -> int:
    if hasattr(dist, "get_global_rank"):
        return int(dist.get_global_rank(group, int(group_rank)))
    return int(dist.get_process_group_ranks(group)[int(group_rank)])


def _ring_rotate_equal(
    tensor: torch.Tensor,
    *,
    group,
    rank: int,
    size: int,
) -> list[torch.Tensor]:
    """Collect equal-shaped tensors with batched P2P ring rotations."""

    if int(size) == 1:
        return [tensor]
    next_peer = _group_global_rank(group, (int(rank) + 1) % int(size))
    previous_peer = _group_global_rank(group, (int(rank) - 1) % int(size))
    current = tensor.contiguous()
    shards: list[torch.Tensor | None] = [None] * int(size)
    shards[int(rank)] = current
    for step in range(1, int(size)):
        received = torch.empty_like(current)
        requests = dist.batch_isend_irecv(
            [
                dist.P2POp(dist.isend, current, next_peer, group),
                dist.P2POp(dist.irecv, received, previous_peer, group),
            ]
        )
        for request in requests:
            request.wait()
        origin = (int(rank) - step) % int(size)
        shards[origin] = received
        current = received
    return [shard for shard in shards if shard is not None]


def _ring_all_gather_sequence(
    tensor: torch.Tensor,
    *,
    group,
    rank: int,
    size: int,
    lengths: list[int] | None = None,
) -> tuple[torch.Tensor, list[int]]:
    """Ring-gather a variable-length tensor along dimension one."""

    local_length = int(tensor.shape[1])
    if lengths is None:
        length_tensor = torch.tensor(
            [local_length], dtype=torch.int64, device=tensor.device
        )
        gathered_lengths = [torch.empty_like(length_tensor) for _ in range(int(size))]
        dist.all_gather(gathered_lengths, length_tensor, group=group)
        lengths = [int(value.item()) for value in gathered_lengths]
    if int(lengths[int(rank)]) != local_length:
        raise RuntimeError(
            "Ring-gather length metadata does not match the local tensor: "
            f"{lengths[int(rank)]} != {local_length}."
        )
    maximum = max(lengths, default=0)
    if maximum == 0:
        empty_shape = list(tensor.shape)
        empty_shape[1] = 0
        return tensor.new_empty(empty_shape), lengths
    padded_shape = list(tensor.shape)
    padded_shape[1] = maximum
    padded = tensor.new_zeros(padded_shape)
    if local_length:
        padded[:, :local_length].copy_(tensor)
    padded_shards = _ring_rotate_equal(
        padded,
        group=group,
        rank=rank,
        size=size,
    )
    return torch.cat(
        [shard[:, :length] for shard, length in zip(padded_shards, lengths)],
        dim=1,
    ), lengths


def _sequence_partitions(sequence_length: int, size: int) -> list[tuple[int, int]]:
    return [
        compute_context_parallel_range(
            sequence_length=int(sequence_length),
            context_parallel_rank=rank,
            context_parallel_size=int(size),
        )
        for rank in range(int(size))
    ]


def _ring_context_halo(
    hidden_states: torch.Tensor,
    *,
    local_start: int,
    sequence_length: int,
    radius: int,
    group,
    rank: int,
    size: int,
) -> tuple[torch.Tensor, int, int]:
    """Return ``left + local + right`` without gathering the full long context."""

    radius = max(int(radius), 0)
    partitions = _sequence_partitions(sequence_length, size)
    expected_start, expected_end = partitions[int(rank)]
    local_length = int(hidden_states.shape[1])
    if (
        int(local_start) != expected_start
        or local_length != expected_end - expected_start
    ):
        raise RuntimeError(
            "CP hidden shard does not match the configured partition: "
            f"expected [{expected_start}, {expected_end}), got "
            f"[{local_start}, {local_start + local_length})."
        )
    if radius == 0:
        return hidden_states, int(local_start), int(local_start + local_length)

    minimum_shard = min(end - start for start, end in partitions)
    if minimum_shard < radius:
        # Tiny smoke inputs can span more than one neighbor.  A full ring gather
        # is safe here because the entire sequence is smaller than size*radius;
        # the 128K path always takes the boundary-only branch below.
        full_hidden, _ = _ring_all_gather_sequence(
            hidden_states,
            group=group,
            rank=rank,
            size=size,
        )
        extended_start = max(0, int(local_start) - radius)
        extended_end = min(
            int(sequence_length), int(local_start) + local_length + radius
        )
        return (
            full_hidden[:, extended_start:extended_end],
            extended_start,
            extended_end,
        )

    boundary = torch.cat(
        [hidden_states[:, :radius], hidden_states[:, -radius:]], dim=1
    ).contiguous()
    boundaries = _ring_rotate_equal(
        boundary,
        group=group,
        rank=rank,
        size=size,
    )
    pieces = []
    extended_start = int(local_start)
    if int(rank) > 0:
        pieces.append(boundaries[int(rank) - 1][:, radius:])
        extended_start -= radius
    pieces.append(hidden_states)
    extended_end = int(local_start) + local_length
    if int(rank) + 1 < int(size):
        pieces.append(boundaries[int(rank) + 1][:, :radius])
        extended_end += radius
    return torch.cat(pieces, dim=1), extended_start, extended_end


def ring_left_context(
    hidden_states: torch.Tensor,
    *,
    local_start: int,
    sequence_length: int,
    window: int,
    group,
    rank: int,
    size: int,
) -> tuple[torch.Tensor, int]:
    """Return the preceding ring halo used by the sliding-window draft model."""

    extended, extended_start, _ = _ring_context_halo(
        hidden_states,
        local_start=local_start,
        sequence_length=sequence_length,
        radius=int(window),
        group=group,
        rank=rank,
        size=size,
    )
    halo_length = int(local_start) - int(extended_start)
    return extended[:, :halo_length], int(extended_start)


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


def _window_ids(
    *, local_start: int, local_end: int, sequence_length: int, ratio: int, device
) -> torch.Tensor:
    first = (int(local_start) + int(ratio) - 1) // int(ratio)
    stop = min(
        (int(local_end) + int(ratio) - 1) // int(ratio),
        int(sequence_length) // int(ratio),
    )
    return torch.arange(first, stop, dtype=torch.long, device=device)


def _gather_token_windows(
    tensor: torch.Tensor,
    *,
    window_ids: torch.Tensor,
    ratio: int,
    extended_start: int,
) -> torch.Tensor:
    offsets = torch.arange(int(ratio), device=tensor.device)
    indices = window_ids[:, None] * int(ratio) + offsets[None, :]
    local_indices = indices - int(extended_start)
    if local_indices.numel() and (
        int(local_indices.min()) < 0
        or int(local_indices.max()) >= int(tensor.shape[1])
    ):
        raise RuntimeError(
            "DeepSeek-V4 CP compressor window escaped its boundary halo."
        )
    return tensor[:, local_indices]


def _compress_hca_local(
    compressor,
    extended_hidden: torch.Tensor,
    *,
    extended_start: int,
    local_start: int,
    local_end: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = int(compressor.compress_rate)
    window_ids = _window_ids(
        local_start=local_start,
        local_end=local_end,
        sequence_length=sequence_length,
        ratio=ratio,
        device=extended_hidden.device,
    )
    raw_kv = compressor.kv_proj(extended_hidden)
    raw_gate = compressor.gate_proj(extended_hidden)
    kv = _gather_token_windows(
        raw_kv,
        window_ids=window_ids,
        ratio=ratio,
        extended_start=extended_start,
    )
    gate = _gather_token_windows(
        raw_gate,
        window_ids=window_ids,
        ratio=ratio,
        extended_start=extended_start,
    )
    if int(window_ids.numel()) == 0:
        return raw_kv.new_zeros((raw_kv.shape[0], 0, compressor.head_dim)), window_ids
    gate = gate + compressor.position_bias.view(1, 1, ratio, -1)
    compressed = compressor.kv_norm(
        (kv * gate.softmax(dim=2, dtype=torch.float32).to(kv.dtype)).sum(dim=2)
    )
    positions = (window_ids * ratio).unsqueeze(0).expand(compressed.shape[0], -1)
    cos, sin = compressor.rotary_emb(
        compressed, position_ids=positions, layer_type=compressor.rope_layer_type
    )
    compressed = _apply_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
    return compressed, window_ids * ratio


def _compress_csa_local(
    compressor,
    extended_hidden: torch.Tensor,
    *,
    extended_start: int,
    local_start: int,
    local_end: int,
    sequence_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ratio = int(compressor.compress_rate)
    head_dim = int(compressor.head_dim)
    window_ids = _window_ids(
        local_start=local_start,
        local_end=local_end,
        sequence_length=sequence_length,
        ratio=ratio,
        device=extended_hidden.device,
    )
    raw_kv = compressor.kv_proj(extended_hidden)
    raw_gate = compressor.gate_proj(extended_hidden)
    if int(window_ids.numel()) == 0:
        return raw_kv.new_zeros((raw_kv.shape[0], 0, head_dim)), window_ids

    current_kv = _gather_token_windows(
        raw_kv,
        window_ids=window_ids,
        ratio=ratio,
        extended_start=extended_start,
    )
    current_gate = _gather_token_windows(
        raw_gate,
        window_ids=window_ids,
        ratio=ratio,
        extended_start=extended_start,
    )
    current_gate = current_gate + compressor.position_bias.view(
        1, 1, ratio, 2 * head_dim
    )
    combined_kv = raw_kv.new_zeros(
        (raw_kv.shape[0], window_ids.numel(), 2 * ratio, head_dim)
    )
    combined_gate = raw_gate.new_full(
        (raw_gate.shape[0], window_ids.numel(), 2 * ratio, head_dim),
        float("-inf"),
    )
    combined_kv[:, :, ratio:] = current_kv[..., head_dim:]
    combined_gate[:, :, ratio:] = current_gate[..., head_dim:]

    has_previous = window_ids > 0
    if bool(has_previous.any()):
        previous_ids = window_ids[has_previous] - 1
        previous_kv = _gather_token_windows(
            raw_kv,
            window_ids=previous_ids,
            ratio=ratio,
            extended_start=extended_start,
        )
        previous_gate = _gather_token_windows(
            raw_gate,
            window_ids=previous_ids,
            ratio=ratio,
            extended_start=extended_start,
        )
        previous_gate = previous_gate + compressor.position_bias.view(
            1, 1, ratio, 2 * head_dim
        )
        combined_kv[:, has_previous, :ratio] = previous_kv[..., :head_dim]
        combined_gate[:, has_previous, :ratio] = previous_gate[..., :head_dim]

    compressed = compressor.kv_norm(
        (
            combined_kv
            * combined_gate.softmax(dim=2, dtype=torch.float32).to(
                combined_kv.dtype
            )
        ).sum(dim=2)
    )
    positions = (window_ids * ratio).unsqueeze(0).expand(compressed.shape[0], -1)
    cos, sin = compressor.rotary_emb(
        compressed, position_ids=positions, layer_type=compressor.rope_layer_type
    )
    compressed = _apply_rotary(compressed.unsqueeze(1), cos, sin).squeeze(1)
    return compressed, window_ids * ratio


def _chunked_csa_topk(
    *,
    indexer,
    hidden_states: torch.Tensor,
    q_residual: torch.Tensor,
    position_ids: torch.Tensor,
    compressed_index_kv: torch.Tensor,
    compressed_positions: torch.Tensor,
    tensor_parallel_group=None,
    tensor_parallel_size: int = 1,
) -> torch.Tensor:
    batch, sequence_length = hidden_states.shape[:2]
    total_keys = int(compressed_index_kv.shape[1])
    top_k = min(int(indexer.index_topk), total_keys)
    if top_k == 0:
        return torch.empty(
            (batch, sequence_length, 0),
            dtype=torch.long,
            device=hidden_states.device,
        )

    cos_q, sin_q = indexer.rotary_emb(
        hidden_states,
        position_ids=position_ids,
        layer_type=indexer.rope_layer_type,
    )
    q = indexer.q_b_proj(q_residual).view(
        batch, sequence_length, indexer.num_heads, indexer.head_dim
    )
    q = _apply_rotary(q.transpose(1, 2), cos_q, sin_q).transpose(1, 2)
    weights = (
        indexer.scorer.weights_proj(hidden_states).float()
        * float(indexer.scorer.weights_scaling)
    )
    q_chunk = max(int(os.getenv("DEEPSPEC_V4_CP_QUERY_CHUNK", "16")), 1)
    key_chunk = max(int(os.getenv("DEEPSPEC_V4_CP_INDEX_KEY_CHUNK", "1024")), 1)
    ratio = int(indexer.compress_rate)
    outputs = []
    for query_start in range(0, sequence_length, q_chunk):
        query_end = min(query_start + q_chunk, sequence_length)
        query = q[:, query_start:query_end].float()
        query_weights = weights[:, query_start:query_end]
        query_positions = position_ids[:, query_start:query_end]
        best_scores = None
        best_indices = None
        for key_start in range(0, total_keys, key_chunk):
            key_end = min(key_start + key_chunk, total_keys)
            keys = compressed_index_kv[:, key_start:key_end].float()
            scores = torch.einsum("bqhd,bkd->bqhk", query, keys)
            scores = F.relu(scores) * float(indexer.scorer.softmax_scale)
            scores = (scores * query_weights.unsqueeze(-1)).sum(dim=2)
            if int(tensor_parallel_size) > 1:
                dist.all_reduce(
                    scores,
                    op=dist.ReduceOp.SUM,
                    group=tensor_parallel_group,
                )
            key_positions = compressed_positions[:, key_start:key_end]
            legal = (
                key_positions[:, None, :] + ratio
                <= query_positions[:, :, None] + 1
            )
            scores = scores.masked_fill(~legal, float("-inf"))
            block_k = min(top_k, key_end - key_start)
            block_scores, block_indices = scores.topk(block_k, dim=-1)
            block_indices = block_indices + key_start
            if best_scores is None:
                candidates_scores = block_scores
                candidates_indices = block_indices
            else:
                candidates_scores = torch.cat(
                    [best_scores, block_scores], dim=-1
                )
                candidates_indices = torch.cat(
                    [best_indices, block_indices], dim=-1
                )
            keep = min(top_k, int(candidates_scores.shape[-1]))
            best_scores, selected = candidates_scores.topk(keep, dim=-1)
            best_indices = torch.gather(candidates_indices, -1, selected)
        best_indices = torch.where(
            torch.isfinite(best_scores),
            best_indices,
            torch.full_like(best_indices, -1),
        )
        outputs.append(best_indices)
    return torch.cat(outputs, dim=1)


def _batched_gather(pool: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    batch, query_length, selected = indices.shape
    if selected == 0:
        return pool.new_empty((batch, query_length, 0, pool.shape[-1]))
    safe = indices.clamp(min=0, max=max(int(pool.shape[1]) - 1, 0))
    expanded = pool[:, None].expand(-1, query_length, -1, -1)
    return torch.gather(
        expanded,
        2,
        safe.unsqueeze(-1).expand(-1, -1, -1, pool.shape[-1]),
    )


@dataclass
class _CPRuntime:
    group: object
    rank: int
    size: int
    sequence_length: int
    local_start: int
    local_end: int
    rotary_emb: object
    tensor_parallel_group: object | None
    tensor_parallel_rank: int
    tensor_parallel_size: int


def _deepseek_v4_cp_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings,
    position_ids: torch.Tensor,
    attention_mask=None,
    past_key_values=None,
    **kwargs,
):
    runtime: _CPRuntime | None = getattr(self, "_deepspec_cp_runtime", None)
    if runtime is None:
        return self._deepspec_original_forward(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            **kwargs,
        )
    del attention_mask, past_key_values, kwargs

    local_length = int(hidden_states.shape[1])
    radius = max(int(self.sliding_window), 128)
    extended_hidden, extended_start, _ = _ring_context_halo(
        hidden_states,
        local_start=runtime.local_start,
        sequence_length=runtime.sequence_length,
        radius=radius,
        group=runtime.group,
        rank=runtime.rank,
        size=runtime.size,
    )
    local_cos, local_sin = position_embeddings[self.rope_layer_type]
    q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
    q = self.q_b_proj(q_residual).view(
        hidden_states.shape[0], local_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    q = self.q_b_norm(q)
    q = _apply_rotary(q, local_cos, local_sin)

    extended_positions = torch.arange(
        extended_start,
        extended_start + int(extended_hidden.shape[1]),
        device=hidden_states.device,
        dtype=torch.long,
    ).unsqueeze(0)
    ext_cos, ext_sin = runtime.rotary_emb(
        extended_hidden,
        position_ids=extended_positions,
        layer_type=self.rope_layer_type,
    )
    extended_kv = self.kv_norm(self.kv_proj(extended_hidden)).unsqueeze(1)
    extended_kv = _apply_rotary(extended_kv, ext_cos, ext_sin)

    compressed_kv = None
    compressed_positions = None
    selected_indices = None
    compression_ratio = None
    if self.compressor is not None:
        compression_ratio = int(self.compressor.compress_rate)
        if self.layer_type == "compressed_sparse_attention":
            local_compressed, local_positions = _compress_csa_local(
                self.compressor,
                extended_hidden,
                extended_start=extended_start,
                local_start=runtime.local_start,
                local_end=runtime.local_end,
                sequence_length=runtime.sequence_length,
            )
            local_index, _ = _compress_csa_local(
                self.compressor.indexer,
                extended_hidden,
                extended_start=extended_start,
                local_start=runtime.local_start,
                local_end=runtime.local_end,
                sequence_length=runtime.sequence_length,
            )
        elif self.layer_type == "heavily_compressed_attention":
            local_compressed, local_positions = _compress_hca_local(
                self.compressor,
                extended_hidden,
                extended_start=extended_start,
                local_start=runtime.local_start,
                local_end=runtime.local_end,
                sequence_length=runtime.sequence_length,
            )
            local_index = None
        else:
            raise ValueError(f"Unsupported DeepSeek-V4 layer type {self.layer_type!r}.")

        compressed_kv, compressed_lengths = _ring_all_gather_sequence(
            local_compressed,
            group=runtime.group,
            rank=runtime.rank,
            size=runtime.size,
        )
        local_position_tensor = local_positions.unsqueeze(0)
        compressed_positions, _ = _ring_all_gather_sequence(
            local_position_tensor,
            group=runtime.group,
            rank=runtime.rank,
            size=runtime.size,
            lengths=compressed_lengths,
        )
        if local_index is not None:
            compressed_index, _ = _ring_all_gather_sequence(
                local_index,
                group=runtime.group,
                rank=runtime.rank,
                size=runtime.size,
                lengths=compressed_lengths,
            )
            selected_indices = _chunked_csa_topk(
                indexer=self.compressor.indexer,
                hidden_states=hidden_states,
                q_residual=q_residual,
                position_ids=position_ids,
                compressed_index_kv=compressed_index,
                compressed_positions=compressed_positions,
                tensor_parallel_group=runtime.tensor_parallel_group,
                tensor_parallel_size=runtime.tensor_parallel_size,
            )

    output_chunks = []
    query_chunk = max(int(os.getenv("DEEPSPEC_V4_CP_QUERY_CHUNK", "16")), 1)
    window = int(self.sliding_window)
    kv_base = extended_kv[:, 0]
    for query_start in range(0, local_length, query_chunk):
        query_end = min(query_start + query_chunk, local_length)
        query_positions = position_ids[:, query_start:query_end]
        offsets = torch.arange(window, device=hidden_states.device)
        sliding_positions = query_positions[:, :, None] - window + 1 + offsets
        sliding_valid = (sliding_positions >= 0) & (
            sliding_positions <= query_positions[:, :, None]
        )
        sliding_indices = (sliding_positions - extended_start).clamp(
            min=0, max=int(kv_base.shape[1]) - 1
        )
        sliding_kv = _batched_gather(kv_base, sliding_indices)

        candidate_kv = [sliding_kv]
        candidate_valid = [sliding_valid]
        if compressed_kv is not None and int(compressed_kv.shape[1]) > 0:
            if selected_indices is not None:
                chosen = selected_indices[:, query_start:query_end]
                candidate_kv.append(_batched_gather(compressed_kv, chosen))
                candidate_valid.append(chosen >= 0)
            else:
                expanded = compressed_kv[:, None].expand(
                    -1, query_end - query_start, -1, -1
                )
                legal = (
                    compressed_positions[:, None, :] + int(compression_ratio)
                    <= query_positions[:, :, None] + 1
                )
                candidate_kv.append(expanded)
                candidate_valid.append(legal)
        candidates = torch.cat(candidate_kv, dim=2)
        valid = torch.cat(candidate_valid, dim=2)
        query = q[:, :, query_start:query_end]
        logits = torch.einsum("bhqd,bqkd->bhqk", query, candidates)
        logits = logits * float(self.scaling)
        logits = logits.masked_fill(~valid.unsqueeze(1), float("-inf"))
        sinks = self.sinks.view(1, -1, 1, 1).expand(
            logits.shape[0], -1, logits.shape[2], -1
        )
        combined = torch.cat([logits, sinks], dim=-1)
        combined = combined - combined.max(dim=-1, keepdim=True).values
        probabilities = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1]
        probabilities = probabilities.to(candidates.dtype)
        output_chunks.append(
            torch.einsum("bhqk,bqkd->bhqd", probabilities, candidates)
        )
    attn_output = torch.cat(output_chunks, dim=2)
    attn_output = _apply_rotary(attn_output, local_cos, -local_sin)
    local_o_groups = int(
        getattr(self, "_deepspec_local_o_groups", self.config.o_groups)
    )
    grouped = attn_output.transpose(1, 2).reshape(
        hidden_states.shape[0], local_length, local_o_groups, -1
    )
    grouped = self.o_a_proj(grouped).flatten(2)
    return self.o_b_proj(grouped), None


def _get_backbone(target_model):
    candidate = getattr(target_model, "model", target_model)
    if hasattr(candidate, "language_model"):
        candidate = candidate.language_model
    if not hasattr(candidate, "layers"):
        raise TypeError("DeepSeek-V4 target model does not expose decoder layers.")
    return candidate


def _unwrap_layer(layer):
    return getattr(layer, "module", layer)


def _deepseek_v4_forward_context_parallel(
    self,
    *,
    model_inputs,
    target_layer_ids,
    context_parallel_group,
    context_parallel_rank: int,
    context_parallel_size: int,
    tensor_parallel_group=None,
    tensor_parallel_rank: int = 0,
    tensor_parallel_size: int = 1,
    device,
):
    del device
    input_ids = model_inputs["input_ids"]
    if int(input_ids.shape[0]) != 1:
        raise ValueError("DeepSeek-V4 Ring CP currently requires batch size 1.")
    attention_mask = model_inputs["attention_mask"]
    if attention_mask.shape != input_ids.shape:
        raise ValueError("DeepSeek-V4 Ring CP input IDs and attention mask must align.")
    valid_sequence_length = int(attention_mask.sum(dim=1)[0].item())
    if valid_sequence_length < 1:
        raise ValueError("DeepSeek-V4 Ring CP received an empty token sequence.")
    if not bool(attention_mask[0, :valid_sequence_length].all()) or bool(
        attention_mask[0, valid_sequence_length:].any()
    ):
        raise ValueError("DeepSeek-V4 Ring CP requires right-padded attention masks.")
    # CP partitions the aligned tensor shape, not the number of valid tokens.
    # Right padding is causally invisible to the real prefix and guarantees
    # every rank owns a multiple of the model's 128-token attention unit.
    sequence_length = int(input_ids.shape[1])
    local_start, local_end = compute_context_parallel_range(
        sequence_length=sequence_length,
        context_parallel_rank=int(context_parallel_rank),
        context_parallel_size=int(context_parallel_size),
    )
    backbone = _get_backbone(self)
    local_input_ids = input_ids[:, local_start:local_end]
    inputs_embeds = backbone.embed_tokens(local_input_ids)
    position_ids = torch.arange(
        local_start,
        local_end,
        dtype=torch.long,
        device=inputs_embeds.device,
    ).unsqueeze(0)
    hidden_streams = inputs_embeds.unsqueeze(2).expand(
        -1, -1, int(backbone.config.hc_mult), -1
    ).contiguous()
    position_embeddings = {
        "main": backbone.rotary_emb(
            inputs_embeds, position_ids=position_ids, layer_type="main"
        ),
        "compress": backbone.rotary_emb(
            inputs_embeds, position_ids=position_ids, layer_type="compress"
        ),
    }
    requested = [int(layer_id) for layer_id in target_layer_ids]
    captured = {}
    if -1 in requested:
        captured[-1] = inputs_embeds.detach()
    runtime = _CPRuntime(
        group=context_parallel_group,
        rank=int(context_parallel_rank),
        size=int(context_parallel_size),
        sequence_length=sequence_length,
        local_start=local_start,
        local_end=local_end,
        rotary_emb=backbone.rotary_emb,
        tensor_parallel_group=tensor_parallel_group,
        tensor_parallel_rank=int(tensor_parallel_rank),
        tensor_parallel_size=int(tensor_parallel_size),
    )
    for layer_idx, layer in enumerate(backbone.layers):
        unwrapped = _unwrap_layer(layer)
        unwrapped.self_attn._deepspec_cp_runtime = runtime
        try:
            hidden_streams = layer(
                hidden_streams,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                attention_mask=None,
                input_ids=local_input_ids,
                past_key_values=None,
            )
        finally:
            unwrapped.self_attn._deepspec_cp_runtime = None
        if layer_idx in requested:
            captured[layer_idx] = backbone.hc_head(hidden_streams).detach()
    missing = [layer_id for layer_id in requested if layer_id not in captured]
    if missing:
        raise RuntimeError(f"DeepSeek-V4 target layers were not captured: {missing}.")
    final_hidden = backbone.norm(backbone.hc_head(hidden_streams)).detach()
    return TargetForwardResult(
        target_hidden_states=torch.cat(
            [captured[layer_id] for layer_id in requested], dim=-1
        ),
        target_last_hidden_states=final_hidden,
        context_start=local_start,
    )


def install_deepseek_v4_ring_context_parallel(target_model):
    """Attach the model-native Ring CP entry point before FSDP wrapping."""

    if str(target_model.config.model_type) != "deepseek_v4":
        raise ValueError(
            "DeepSeek-V4 CP adapter received model_type="
            f"{target_model.config.model_type!r}."
        )
    if getattr(target_model, "_deepspec_ring_cp_installed", False):
        return target_model
    backbone = _get_backbone(target_model)
    for layer in backbone.layers:
        attention = layer.self_attn
        attention._deepspec_original_forward = attention.forward
        attention._deepspec_cp_runtime = None
        attention.forward = MethodType(
            _deepseek_v4_cp_attention_forward, attention
        )
    target_model.forward_context_parallel = MethodType(
        _deepseek_v4_forward_context_parallel, target_model
    )
    target_model._deepspec_context_layout = "contiguous"
    target_model._deepspec_ring_cp_installed = True
    return target_model


def install_target_context_parallel(target_model):
    model_type = str(target_model.config.model_type)
    if model_type == "deepseek_v4":
        return install_deepseek_v4_ring_context_parallel(target_model)
    if model_type.lower() in ("qwen3_5", "qwen3_5_text"):
        from .qwen3_6_cp import install_qwen3_6_ring_context_parallel

        return install_qwen3_6_ring_context_parallel(target_model)
    raise NotImplementedError(
        f"No model-native target Context Parallel adapter for {model_type!r}."
    )


__all__ = [
    "install_deepseek_v4_ring_context_parallel",
    "install_target_context_parallel",
    "ring_left_context",
]
