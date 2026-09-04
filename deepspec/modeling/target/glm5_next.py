"""Memory-bounded GLM-5.3 target prefill used by DSpark training."""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec
from types import MethodType

import torch
import torch.nn.functional as F
from packaging.version import Version
from transformers.models.glm5_next.modeling_glm5_next import causal_conv1d_fn


def _positive_chunk_size(name: str, default: int) -> int:
    raw_value = os.environ.get(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}.") from error
    if value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return value


def _require_full_prefill_mask(attention_mask: torch.Tensor | None) -> None:
    if (
        attention_mask is None
        or attention_mask.ndim != 2
        or not bool(attention_mask.all().item())
    ):
        raise ValueError(
            "Bounded GLM-5.3 target DSA requires the unpadded 2D mask emitted "
            "by Glm5NextOnlineTarget."
        )


def _l2norm_float(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / torch.sqrt(
        (tensor * tensor).sum(dim=-1, keepdim=True) + 1e-6
    )


def _causal_conv1d_prefill(
    tensor: torch.Tensor,
    *,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    **kwargs,
) -> torch.Tensor:
    """Use the CUDA extension on GPU and its exact depthwise form on CPU."""

    if tensor.is_cuda:
        return causal_conv1d_fn(
            tensor,
            weight=weight,
            bias=bias,
            activation=activation,
            **kwargs,
        )
    output = F.conv1d(
        tensor,
        weight.unsqueeze(1),
        bias=bias,
        padding=int(weight.shape[-1]) - 1,
        groups=int(tensor.shape[1]),
    )[..., : int(tensor.shape[-1])]
    if activation in ("silu", "swish"):
        output = F.silu(output)
    elif activation not in (None, "identity"):
        raise ValueError(f"Unsupported causal-conv activation: {activation!r}.")
    return output


@torch.no_grad()
def _bounded_chunk_kimi_delta_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
):
    """Evaluate the exact KDA chunk recurrence with bounded workspace.

    Transformers' fallback expands the per-channel decay tensor for every
    sequence chunk at once. At 128K this single FP32 tensor is 64 GiB even
    after TP4. Batching a fixed number of independent intra-chunk transforms
    preserves its recurrence while bounding that workspace.
    """

    if chunk_size != 64:
        raise ValueError(
            "Bounded GLM-5.3 KDA preserves the checkpoint's chunk_size=64, "
            f"got {chunk_size}."
        )
    input_dtype = query.dtype
    batch_size, sequence_length, num_heads, key_head_dim = key.shape
    value_head_dim = int(value.shape[-1])
    scale = float(query.shape[-1]) ** -0.5
    chunks_per_batch = _positive_chunk_size(
        "DEEPSPEC_GLM_KDA_CHUNKS_PER_BATCH",
        8,
    )
    token_batch_size = chunk_size * chunks_per_batch

    recurrent_state = (
        torch.zeros(
            batch_size,
            num_heads,
            key_head_dim,
            value_head_dim,
            dtype=torch.float32,
            device=value.device,
        )
        if initial_state is None
        else initial_state.to(device=value.device, dtype=torch.float32)
    )
    output = torch.empty(
        batch_size,
        sequence_length,
        num_heads,
        value_head_dim,
        dtype=input_dtype,
        device=value.device,
    )
    solve_mask = torch.triu(
        torch.ones(
            chunk_size,
            chunk_size,
            dtype=torch.bool,
            device=query.device,
        ),
        diagonal=0,
    )
    causal_mask = torch.triu(solve_mask, diagonal=1)
    identity = torch.eye(
        chunk_size,
        dtype=torch.float32,
        device=query.device,
    )

    def prepare_vector_block(tensor, start, end, pad_size):
        block = tensor[:, start:end].transpose(1, 2).contiguous().float()
        return F.pad(block, (0, 0, 0, pad_size))

    for token_start in range(0, sequence_length, token_batch_size):
        token_end = min(token_start + token_batch_size, sequence_length)
        block_length = token_end - token_start
        pad_size = (-block_length) % chunk_size

        query_block = prepare_vector_block(
            query,
            token_start,
            token_end,
            pad_size,
        )
        key_block = prepare_vector_block(
            key,
            token_start,
            token_end,
            pad_size,
        )
        value_block = prepare_vector_block(
            value,
            token_start,
            token_end,
            pad_size,
        )
        gate_block = prepare_vector_block(
            g,
            token_start,
            token_end,
            pad_size,
        )
        beta_block = beta[:, token_start:token_end].transpose(1, 2)
        beta_block = F.pad(beta_block.float(), (0, pad_size))

        if use_qk_l2norm_in_kernel:
            query_block = _l2norm_float(query_block)
            key_block = _l2norm_float(key_block)
        query_block = query_block * scale

        chunk_count = (block_length + pad_size) // chunk_size
        vector_shape = (
            batch_size,
            num_heads,
            chunk_count,
            chunk_size,
            key_head_dim,
        )
        query_block = query_block.reshape(vector_shape)
        key_block = key_block.reshape(vector_shape)
        value_block = value_block.reshape(
            batch_size,
            num_heads,
            chunk_count,
            chunk_size,
            value_head_dim,
        )
        gate_block = gate_block.reshape(vector_shape).cumsum(dim=-2)
        beta_block = beta_block.reshape(
            batch_size,
            num_heads,
            chunk_count,
            chunk_size,
        )

        value_beta = value_block * beta_block.unsqueeze(-1)
        key_beta = key_block * beta_block.unsqueeze(-1)
        decay = (
            gate_block.unsqueeze(-2) - gate_block.unsqueeze(-3)
        ).exp()
        transition = -(
            key_beta.unsqueeze(-2)
            * key_block.unsqueeze(-3)
            * decay
        ).sum(dim=-1)
        transition.masked_fill_(solve_mask, 0)
        for row_index in range(1, chunk_size):
            row = transition[..., row_index, :row_index].clone()
            lower = transition[..., :row_index, :row_index].clone()
            transition[..., row_index, :row_index] = row + (
                row.unsqueeze(-1) * lower
            ).sum(dim=-2)

        transition = transition + identity
        transformed_value = transition @ value_beta
        cumulative_key = transition @ (key_beta * gate_block.exp())

        for chunk_index in range(chunk_count):
            query_chunk = query_block[:, :, chunk_index]
            key_chunk = key_block[:, :, chunk_index]
            gate_chunk = gate_block[:, :, chunk_index]
            decay_chunk = decay[:, :, chunk_index]
            value_chunk = transformed_value[:, :, chunk_index]
            cumulative_key_chunk = cumulative_key[:, :, chunk_index]

            inter_chunk = (
                query_chunk * gate_chunk.exp()
            ) @ recurrent_state
            intra_chunk = (
                query_chunk.unsqueeze(-2)
                * key_chunk.unsqueeze(-3)
                * decay_chunk
            ).sum(dim=-1)
            intra_chunk.masked_fill_(causal_mask, 0)
            value_prime = cumulative_key_chunk @ recurrent_state
            value_new = value_chunk - value_prime
            chunk_output = inter_chunk + intra_chunk @ value_new

            global_start = token_start + chunk_index * chunk_size
            valid_length = min(
                chunk_size,
                sequence_length - global_start,
            )
            output[:, global_start : global_start + valid_length] = (
                chunk_output[:, :, :valid_length]
                .transpose(1, 2)
                .to(input_dtype)
            )
            recurrent_state = (
                recurrent_state
                * gate_chunk[:, :, -1].exp().unsqueeze(-1)
                + (
                    key_chunk
                    * (
                        gate_chunk[:, :, -1:] - gate_chunk
                    ).exp()
                ).transpose(-1, -2)
                @ value_new
            )

    return output, recurrent_state if output_final_state else None


@torch.no_grad()
def _bounded_linear_attention_forward(
    self,
    hidden_states: torch.Tensor,
    cache_params=None,
    attention_mask: torch.Tensor | None = None,
    **kwargs,
):
    """Run full-prefill GLM KDA without sequence-sized chunk workspace."""

    if cache_params is not None:
        raise NotImplementedError(
            "Bounded GLM-5.3 target KDA supports full prefill only."
        )
    _require_full_prefill_mask(attention_mask)
    batch_size, sequence_length = hidden_states.shape[:2]
    hidden_shape = (
        batch_size,
        sequence_length,
        -1,
        self.head_dim,
    )
    mixed_qkv = torch.cat(
        [
            self.q_proj(hidden_states),
            self.k_proj(hidden_states),
            self.v_proj(hidden_states),
        ],
        dim=-1,
    ).transpose(1, 2)
    mixed_qkv = _causal_conv1d_prefill(
        mixed_qkv,
        weight=self.conv1d.weight.squeeze(1),
        bias=self.conv1d.bias,
        activation=self.activation,
        **kwargs,
    )
    mixed_qkv = mixed_qkv[:, :, -sequence_length:]
    query, key, value = torch.split(
        mixed_qkv.transpose(1, 2),
        [self.qkv_dim] * 3,
        dim=-1,
    )
    query = query.view(hidden_shape)
    key = key.view(hidden_shape)
    value = value.view(hidden_shape)
    gate_decay = self.forget_gate(hidden_states)
    beta = torch.sigmoid(self.b_proj(hidden_states))
    core_output, _ = _bounded_chunk_kimi_delta_attention(
        query,
        key,
        value,
        g=gate_decay,
        beta=beta,
        use_qk_l2norm_in_kernel=True,
    )
    output_gate = self.g_b_proj(self.g_a_proj(hidden_states)).view(
        hidden_shape
    )
    output = self.o_norm(core_output, output_gate).reshape(
        batch_size,
        sequence_length,
        -1,
    )
    return self.o_proj(output)


@torch.no_grad()
def _bounded_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    attention_mask: torch.Tensor,
    past_key_values,
) -> torch.LongTensor:
    """Run exact GLM k-pool selection without a quadratic score tensor."""

    if past_key_values is not None:
        raise NotImplementedError(
            "The bounded GLM-5.3 target indexer supports full prefill only."
        )
    _require_full_prefill_mask(attention_mask)
    batch_size, sequence_length = hidden_states.shape[:2]

    keys = self.k_norm(self.wk(hidden_states)).view(
        batch_size,
        sequence_length,
        self.head_dim,
    )
    gate_scores = F.linear(hidden_states, self.index_kpool_compress_gate)
    valid_channel = torch.ones(
        (batch_size, sequence_length, 1),
        dtype=keys.dtype,
        device=keys.device,
    )
    packed_states = torch.cat([keys, gate_scores, valid_channel], dim=-1)
    pool_keys, pool_indices, pool_valid = self.get_pooled_states(
        packed_states=packed_states
    )
    del packed_states, keys, gate_scores, valid_channel

    pool_count = int(pool_keys.shape[1])
    selected_pool_count = min(
        int(self.index_topk) // int(self.index_kpool),
        pool_count,
    )
    output_width = int(self.index_topk)
    if self.index_kpool_always_select_tail:
        output_width += int(self.index_kpool) - 1
    topk_indices = torch.full(
        (batch_size, sequence_length, output_width),
        -1,
        dtype=torch.int32,
        device=hidden_states.device,
    )

    query_chunk_size = _positive_chunk_size(
        "DEEPSPEC_GLM_INDEX_QUERY_CHUNK",
        128,
    )
    pool_ends = (
        pool_indices[..., -1]
        if pool_count
        else torch.empty(
            (batch_size, 0), dtype=torch.long, device=hidden_states.device
        )
    )
    batch_indices = torch.arange(
        batch_size, device=hidden_states.device
    )[:, None, None]

    for query_start in range(0, sequence_length, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sequence_length)
        chunk_length = query_end - query_start
        query_positions = torch.arange(
            query_start,
            query_end,
            device=hidden_states.device,
        )

        if selected_pool_count:
            query = self.wq_b(q_resid[:, query_start:query_end]).view(
                batch_size,
                chunk_length,
                self.n_heads,
                self.head_dim,
            )
            scores = torch.matmul(
                query.float(),
                pool_keys.transpose(-1, -2).float().unsqueeze(1),
            )
            scores = F.relu(scores * self.softmax_scale)
            weights = self.weights_proj(
                hidden_states[:, query_start:query_end].to(
                    self.weights_proj.weight.dtype
                )
            ).float()
            weights = weights * (self.n_heads**-0.5)
            index_scores = torch.matmul(
                weights.unsqueeze(-2), scores
            ).squeeze(-2)

            valid_candidates = (
                pool_valid[:, None, :]
                & pool_ends[:, None, :].ge(0)
                & pool_ends[:, None, :].le(query_positions[None, :, None])
            )
            index_scores.masked_fill_(
                ~valid_candidates,
                torch.finfo(index_scores.dtype).min,
            )
            selected = index_scores.topk(
                selected_pool_count,
                dim=-1,
            ).indices
            selected_valid = valid_candidates.gather(-1, selected)
            selected_indices = pool_indices[batch_indices, selected]
            selected_indices = selected_indices.masked_fill(
                ~selected_valid[..., None].expand_as(selected_indices),
                -1,
            )
            chunk_indices = selected_indices.flatten(-2)
        else:
            chunk_indices = torch.empty(
                (batch_size, chunk_length, 0),
                dtype=torch.long,
                device=hidden_states.device,
            )

        if self.index_kpool_always_select_tail and self.index_kpool > 1:
            max_tail_width = int(self.index_kpool) - 1
            visible_count = query_positions + 1
            tail_count = visible_count.remainder(int(self.index_kpool))
            tail_offsets = torch.arange(
                max_tail_width,
                device=hidden_states.device,
            )
            tail_start = visible_count - tail_count
            tail = tail_start[:, None] + tail_offsets[None, :]
            tail_valid = tail_offsets[None, :] < tail_count[:, None]
            tail = tail.masked_fill(~tail_valid, -1)
            chunk_indices = torch.cat(
                [
                    chunk_indices,
                    tail[None, :, :].expand(batch_size, -1, -1),
                ],
                dim=-1,
            )

        if int(chunk_indices.shape[-1]) < output_width:
            chunk_indices = F.pad(
                chunk_indices,
                (0, output_width - int(chunk_indices.shape[-1])),
                value=-1,
            )
        topk_indices[:, query_start:query_end] = chunk_indices[
            ..., :output_width
        ].to(torch.int32)

    return topk_indices


def _can_use_native_nope_sparse_mla(self, hidden_states: torch.Tensor) -> bool:
    """Return whether FlashInfer's released GLM NoPE kernel fits this layer."""

    if not hidden_states.is_cuda or not _flashinfer_supports_glm_nope():
        return False
    major, _minor = torch.cuda.get_device_capability(hidden_states.device)
    return (
        major == 10
        and int(self.kv_lora_rank) == 512
        and int(self.qk_rope_head_dim) == 0
        and int(self.qk_nope_head_dim) == 256
        and int(self.v_head_dim) == 256
        and hidden_states.dtype == torch.bfloat16
    )


def _flashinfer_supports_glm_nope() -> bool:
    if find_spec("flashinfer") is None:
        return False
    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError:
        return False
    return Version(installed) >= Version("0.6.18")


def _pack_kpool_topk_for_native_kernel(
    topk_indices: torch.Tensor,
    *,
    query_start: int,
    index_topk: int,
    index_kpool: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack valid pooled and tail indices before the native kernel's -1 pad."""

    chunk_length = int(topk_indices.shape[1])
    output_width = int(topk_indices.shape[2])
    expected_width = index_topk + index_kpool - 1
    if output_width != expected_width:
        raise ValueError(
            "Native GLM DSA expected index_topk + index_kpool - 1 columns, "
            f"got {output_width} instead of {expected_width}."
        )

    positions = torch.arange(
        query_start,
        query_start + chunk_length,
        dtype=torch.long,
        device=topk_indices.device,
    )
    visible = positions + 1
    full_pool_tokens = (
        torch.div(visible, index_kpool, rounding_mode="floor") * index_kpool
    ).clamp(max=index_topk)
    tail_tokens = visible.remainder(index_kpool)
    active_lengths = (full_pool_tokens + tail_tokens).to(torch.int32)

    # The indexer stores selected complete pools in [:index_topk] and the
    # unfinished causal tail in [index_topk:]. Move that tail in front of the
    # -1 padding; FlashInfer requires packed rows but attention is invariant to
    # this permutation of the selected keys.
    columns = torch.arange(
        expected_width,
        dtype=torch.long,
        device=topk_indices.device,
    )[None, :]
    full_pool_tokens = full_pool_tokens[:, None]
    tail_tokens = tail_tokens[:, None]
    tail_columns = index_topk + columns - full_pool_tokens
    source_columns = torch.where(
        columns < full_pool_tokens,
        columns,
        torch.where(
            columns < full_pool_tokens + tail_tokens,
            tail_columns,
            -1,
        ),
    )
    source_columns = source_columns[None, :, :]
    packed = topk_indices.gather(
        -1,
        source_columns.clamp(min=0).expand(int(topk_indices.shape[0]), -1, -1),
    )
    packed = packed.masked_fill(source_columns.lt(0), -1)
    padded_width = ((expected_width + 3) // 4) * 4
    if padded_width != expected_width:
        packed = F.pad(packed, (0, padded_width - expected_width), value=-1)
    return packed.contiguous(), active_lengths


@torch.no_grad()
def _native_nope_sparse_attention(
    self,
    *,
    q_resid: torch.Tensor,
    kv_pass: torch.Tensor,
    topk_indices: torch.Tensor,
) -> torch.Tensor:
    """Run exact selected-token MLA with FlashInfer's SM100 NoPE kernel."""

    import flashinfer.decode

    batch_size, sequence_length = q_resid.shape[:2]
    if batch_size != 1:
        raise ValueError("Native GLM target DSA requires local_batch_size=1.")
    num_heads = int(self.num_heads)
    index_topk = int(self.config.index_topk)
    index_kpool = int(self.config.index_kpool)
    kv_weights = self.kv_b_proj.weight.view(
        num_heads,
        int(self.qk_nope_head_dim) + int(self.v_head_dim),
        int(self.kv_lora_rank),
    )
    key_weight, value_weight = kv_weights.split(
        [int(self.qk_nope_head_dim), int(self.v_head_dim)],
        dim=1,
    )
    value_weight = value_weight.transpose(1, 2).contiguous()
    # TRTLLM-GEN validates a physical page size of 32 or 64 even though its
    # sparse MLA kernels interpret block-table entries as flattened token
    # locations (the sparse kernel itself has an effective page size of one).
    # Use the same 64-token storage layout as SGLang and pad only the final
    # physical page; selected indices continue to address original tokens.
    page_size = 64
    kv_tokens = kv_pass[0, 0]
    page_padding = (-sequence_length) % page_size
    if page_padding:
        kv_tokens = F.pad(kv_tokens, (0, 0, 0, page_padding))
    kv_cache = kv_tokens.view(-1, 1, page_size, int(self.kv_lora_rank))
    output = q_resid.new_empty(
        batch_size,
        sequence_length,
        num_heads,
        int(self.v_head_dim),
    )

    workspace_state = self._deepspec_native_workspace_state
    workspace_bytes = _positive_chunk_size(
        "DEEPSPEC_GLM_DSA_WORKSPACE_BYTES",
        384 * 1024 * 1024,
    )
    workspace_key = (q_resid.device.index, workspace_bytes)
    workspace = workspace_state.get(workspace_key)
    if workspace is None:
        workspace = torch.zeros(
            workspace_bytes,
            dtype=torch.uint8,
            device=q_resid.device,
        )
        workspace_state.clear()
        workspace_state[workspace_key] = workspace

    query_chunk_size = _positive_chunk_size(
        "DEEPSPEC_GLM_NATIVE_ATTN_QUERY_CHUNK",
        4096,
    )
    for query_start in range(0, sequence_length, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sequence_length)
        query = self.q_b_proj(q_resid[:, query_start:query_end]).view(
            query_end - query_start,
            num_heads,
            int(self.qk_nope_head_dim),
        )
        absorbed_query = torch.bmm(
            query.transpose(0, 1),
            key_weight,
        ).transpose(0, 1).contiguous()
        packed_indices, active_lengths = _pack_kpool_topk_for_native_kernel(
            topk_indices[:, query_start:query_end],
            query_start=query_start,
            index_topk=index_topk,
            index_kpool=index_kpool,
        )
        latent_output = flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=absorbed_query.unsqueeze(1),
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            qk_nope_head_dim=int(self.qk_nope_head_dim),
            kv_lora_rank=int(self.kv_lora_rank),
            qk_rope_head_dim=0,
            block_tables=packed_indices[0].unsqueeze(1),
            seq_lens=active_lengths,
            max_seq_len=sequence_length,
            sparse_mla_top_k=int(packed_indices.shape[-1]),
            bmm1_scale=float(self.scaling),
            backend="trtllm-gen",
        )
        latent_output = latent_output.squeeze(1)
        value_output = torch.bmm(
            latent_output.transpose(0, 1),
            value_weight,
        ).transpose(0, 1)
        output[:, query_start:query_end] = value_output.unsqueeze(0)

    self._deepspec_used_native_sparse_attention = True
    return output


def _bounded_sparse_attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: torch.Tensor | None,
    past_key_values=None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs,
):
    """Run selected-token MLA attention without constructing a dense mask."""

    del kwargs
    if past_key_values is not None:
        raise NotImplementedError(
            "Bounded GLM-5.3 target DSA supports full prefill only."
        )
    _require_full_prefill_mask(attention_mask)
    batch_size, sequence_length = hidden_states.shape[:2]

    q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    kv_pass, k_rot = torch.split(
        compressed_kv,
        [self.kv_lora_rank, self.qk_rope_head_dim],
        dim=-1,
    )
    kv_pass = self.kv_a_layernorm(kv_pass).view(
        batch_size,
        1,
        sequence_length,
        self.kv_lora_rank,
    )
    k_rot = k_rot.view(
        batch_size,
        1,
        sequence_length,
        self.qk_rope_head_dim,
    )
    if self.indexer is not None:
        topk_indices = self.indexer(
            hidden_states=hidden_states,
            q_resid=q_resid,
            attention_mask=attention_mask,
            past_key_values=None,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError(
                "Shared DSA layers require top-k indices from a previous full "
                "indexer layer."
            )
        topk_indices = prev_topk_indices

    if _can_use_native_nope_sparse_mla(self, hidden_states):
        output = _native_nope_sparse_attention(
            self,
            q_resid=q_resid,
            kv_pass=kv_pass,
            topk_indices=topk_indices,
        )
        output = output.reshape(batch_size, sequence_length, -1).contiguous()
        output = self.o_proj(output)
        return output, None, topk_indices if self.next_skip_topk else None

    key_states, value_states = self.expand_kv(kv_pass, k_rot)

    key_by_token = key_states.transpose(1, 2)
    value_by_token = value_states.transpose(1, 2)
    output = hidden_states.new_empty(
        batch_size,
        sequence_length,
        self.num_heads,
        self.v_head_dim,
    )
    query_chunk_size = _positive_chunk_size(
        "DEEPSPEC_GLM_ATTN_QUERY_CHUNK",
        32,
    )
    batch_indices = torch.arange(
        batch_size, device=hidden_states.device
    )[:, None, None]

    for query_start in range(0, sequence_length, query_chunk_size):
        query_end = min(query_start + query_chunk_size, sequence_length)
        chunk_length = query_end - query_start
        query_states = self.q_b_proj(
            q_resid[:, query_start:query_end]
        ).view(
            batch_size,
            chunk_length,
            self.num_heads,
            self.qk_head_dim,
        )
        query_states = query_states.transpose(1, 2)

        chunk_topk = topk_indices[:, query_start:query_end].long()
        valid = chunk_topk.ge(0) & chunk_topk.lt(sequence_length)
        safe_indices = chunk_topk.clamp(0, sequence_length - 1)
        selected_keys = key_by_token[batch_indices, safe_indices]
        selected_values = value_by_token[batch_indices, safe_indices]
        selected_keys = selected_keys.permute(0, 3, 1, 2, 4)
        selected_values = selected_values.permute(0, 3, 1, 2, 4)

        scores = torch.matmul(
            query_states.unsqueeze(-2),
            selected_keys.transpose(-1, -2),
        ).squeeze(-2)
        scores = scores * self.scaling
        scores.masked_fill_(
            ~valid[:, None, :, :],
            torch.finfo(scores.dtype).min,
        )
        probabilities = F.softmax(scores, dim=-1, dtype=torch.float32).to(
            query_states.dtype
        )
        probabilities = F.dropout(
            probabilities,
            p=0.0 if not self.training else self.attention_dropout,
            training=self.training,
        )
        chunk_output = torch.matmul(
            probabilities.unsqueeze(-2),
            selected_values,
        ).squeeze(-2)
        output[:, query_start:query_end] = chunk_output.transpose(1, 2)

    output = output.reshape(batch_size, sequence_length, -1).contiguous()
    output = self.o_proj(output)
    return output, None, topk_indices if self.next_skip_topk else None


def _get_glm5_next_text_backbone(model):
    backbone = getattr(model, "model", model)
    backbone = getattr(backbone, "language_model", backbone)
    if str(getattr(getattr(backbone, "config", None), "model_type", "")) != (
        "glm5_next_text"
    ):
        raise ValueError("Expected a GLM-5.3 text backbone.")
    if not hasattr(backbone, "layers"):
        raise TypeError("GLM-5.3 target does not expose decoder layers.")
    return backbone


def install_glm5_next_bounded_target_prefill(model):
    """Install memory-bounded GLM target KDA and DSA prefill paths."""

    if getattr(model, "_deepspec_bounded_glm_target_prefill", False):
        return model
    backbone = _get_glm5_next_text_backbone(model)
    native_workspace_state = {}
    linear_installed = 0
    sparse_installed = 0
    for decoder_layer in backbone.layers:
        attention = decoder_layer.self_attn
        if hasattr(attention, "forget_gate") and hasattr(attention, "conv1d"):
            attention._deepspec_original_forward = attention.forward
            attention.forward = MethodType(
                _bounded_linear_attention_forward,
                attention,
            )
            linear_installed += 1
            continue
        if not hasattr(attention, "kv_b_proj"):
            continue
        attention._deepspec_original_forward = attention.forward
        attention.forward = MethodType(
            _bounded_sparse_attention_forward,
            attention,
        )
        attention._deepspec_native_workspace_state = native_workspace_state
        if attention.indexer is not None:
            attention.indexer._deepspec_original_forward = attention.indexer.forward
            attention.indexer.forward = MethodType(
                _bounded_indexer_forward,
                attention.indexer,
            )
        sparse_installed += 1
    if linear_installed + sparse_installed == 0:
        raise ValueError(
            "GLM-5.3 target contains no supported attention layers."
        )
    model._deepspec_bounded_glm_target_prefill = True
    backbone._deepspec_bounded_glm_target_prefill = True
    return model


def uninstall_glm5_next_bounded_target_prefill(model) -> None:
    """Drop GLM bounded-prefill workspaces and restore replaced bound methods."""

    backbone = _get_glm5_next_text_backbone(model)
    for decoder_layer in backbone.layers:
        attention = decoder_layer.self_attn
        workspace = getattr(attention, "_deepspec_native_workspace_state", None)
        if isinstance(workspace, dict):
            workspace.clear()
        original_forward = getattr(attention, "_deepspec_original_forward", None)
        if original_forward is not None:
            attention.forward = original_forward
            del attention._deepspec_original_forward
        if hasattr(attention, "_deepspec_native_workspace_state"):
            del attention._deepspec_native_workspace_state
        if hasattr(attention, "_deepspec_used_native_sparse_attention"):
            del attention._deepspec_used_native_sparse_attention
        indexer = getattr(attention, "indexer", None)
        if indexer is not None:
            original_forward = getattr(
                indexer, "_deepspec_original_forward", None
            )
            if original_forward is not None:
                indexer.forward = original_forward
                del indexer._deepspec_original_forward
    for owner in (model, backbone):
        if hasattr(owner, "_deepspec_bounded_glm_target_prefill"):
            del owner._deepspec_bounded_glm_target_prefill


__all__ = [
    "install_glm5_next_bounded_target_prefill",
    "uninstall_glm5_next_bounded_target_prefill",
]
