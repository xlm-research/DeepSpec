# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# The compact permutation and dispatch/combine backward design are adapted from
# torchtitan/torchtitan/distributed/deepep/deepep.py (BSD 3-Clause):
# https://github.com/pytorch/torchtitan/tree/main/torchtitan/distributed/deepep
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Draft-owned DeepEP V2 BF16 training communication.

This compact-layout implementation is eager-only: do not include it in CUDA
graphs, torch.compile, or activation checkpoint regions. DeepEP performs a host
sync to determine received token counts. Buffers are created lazily and must be
closed collectively after backward, before unloading the draft. There is no
process-global buffer or handle cache. Multiple layers may share one dispatcher;
each autograd graph retains its own communication handle.
"""

import ctypes
import os
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist
from torch.autograd.function import once_differentiable


def _load_elastic_buffer():
    try:
        from deep_ep import ElasticBuffer, topk_idx_t
    except ImportError as error:
        raise ImportError(
            "Draft expert_dispatch='deepep' requires DeepEP V2 (ElasticBuffer). "
            "Install it in the draft environment: https://github.com/deepseek-ai/DeepEP"
        ) from error
    return ElasticBuffer, topk_idx_t


def _loaded_nccl_version() -> tuple[int, int, int]:
    # torch.cuda.nccl.version() reports the NCCL headers used to build PyTorch,
    # which may differ from a compatible newer runtime in a draft environment.
    # Query the unique library already mapped into this process. RTLD_NOLOAD
    # ensures this check never pulls a second NCCL runtime into the process.
    with open("/proc/self/maps", encoding="utf-8") as maps:
        paths = {
            fields[5]
            for line in maps
            if len(fields := line.rstrip().split(maxsplit=5)) == 6
            and os.path.basename(fields[5]).startswith("libnccl.so")
        }
    if len(paths) != 1:
        raise RuntimeError(
            "DeepEP requires exactly one loaded NCCL shared library; "
            f"found {sorted(paths)}."
        )
    path = paths.pop()
    try:
        nccl = ctypes.CDLL(path, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL)
        get_version = nccl.ncclGetVersion
        get_version.argtypes = [ctypes.POINTER(ctypes.c_int)]
        get_version.restype = ctypes.c_int
        version = ctypes.c_int()
        status = get_version(ctypes.byref(version))
    except (OSError, AttributeError) as error:
        raise RuntimeError(f"Unable to query the loaded NCCL runtime at {path}: {error}") from error
    if status != 0:
        raise RuntimeError(f"Loaded NCCL ncclGetVersion returned error code {status}.")
    return version.value // 10000, version.value % 10000 // 100, version.value % 100


def require_deepep() -> None:
    """Check V2 and the loaded NCCL version without allocating a CUDA buffer."""
    if not dist.is_nccl_available():
        raise RuntimeError("DeepEP V2 requires PyTorch with NCCL support.")
    nccl_version = _loaded_nccl_version()
    if nccl_version < (2, 30, 4):
        raise RuntimeError(
            f"DeepEP V2 requires loaded NCCL >= 2.30.4; found {nccl_version}. "
            "Use a separate compatible draft environment."
        )
    try:
        _load_elastic_buffer()
    except (AssertionError, OSError) as error:
        raise RuntimeError(f"DeepEP V2 dependency initialization failed: {error}") from error


def _wait(event) -> None:
    if event is not None:
        event.current_stream_wait()


@dataclass
class _DispatchState:
    dispatcher: "DeepEPDispatcher"
    handle: object = None


class _Dispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, indices, scores, state):
        dispatcher = state.dispatcher
        recv_hidden, recv_indices, recv_scores, state.handle, event = (
            dispatcher._get_buffer().dispatch(
                hidden,
                topk_idx=indices,
                topk_weights=scores,
                num_experts=dispatcher.num_experts,
                # This is also the global-token-ID stride. All EP ranks must
                # use the configured capacity, even when local T differs.
                num_max_tokens_per_rank=dispatcher.max_tokens_per_rank,
                num_sms=dispatcher._num_sms,
                expert_alignment=1,
                do_expand=False,
                do_cpu_sync=True,
                async_with_compute_stream=True,
            )
        )
        _wait(event)
        ctx.state = state
        ctx.mark_non_differentiable(recv_indices)
        return recv_hidden, recv_indices, recv_scores

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_hidden, _grad_indices, grad_scores):
        # Materialized zero gradients keep all ranks in the same collectives,
        # including a rank receiving no tokens or using only routing gradients.
        grad_input, grad_routing, event = ctx.state.dispatcher._get_buffer().combine(
            grad_hidden.contiguous(),
            handle=ctx.state.handle,
            topk_weights=grad_scores.float().contiguous(),
            async_with_compute_stream=True,
        )
        _wait(event)
        return grad_input, None, grad_routing, None


class _Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, state):
        ctx.state = state
        result, _scores, event = state.dispatcher._get_buffer().combine(
            hidden.contiguous(),
            handle=state.handle,
            # Real routing weights were already applied to expert outputs.
            topk_weights=None,
            async_with_compute_stream=True,
        )
        _wait(event)
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        state = ctx.state
        grad_hidden, _indices, _scores, _handle, event = (
            state.dispatcher._get_buffer().dispatch(
                grad_output.contiguous(),
                handle=state.handle,
                # The V2 heuristic runs before inferring num_experts from a
                # cached handle. Reuse the original SM count explicitly.
                num_sms=state.handle.num_sms,
                do_cpu_sync=False,
                async_with_compute_stream=True,
            )
        )
        _wait(event)
        return grad_hidden, None


class DeepEPDispatcher:
    """Communicate rank-local tokens and run the existing local experts.

    ``expert_forward(tokens, local_indices, unit_weights)`` must be the original
    bound experts forward, captured before installing any parallel wrapper.
    Its input contains one expert-major row per token/expert pair, indices are
    ``[pairs, 1]``, and weights are ones. This dispatcher applies real routing
    weights exactly once before combine. The router's global expert numbering
    must use contiguous equal-sized expert partitions across EP ranks.

    Construction allocates no CUDA memory. The first call collectively verifies
    buffer geometry and allocates it; every rank must call layers/chunks in the
    same order. A shared instance may serve multiple draft layers of equal MoE
    geometry. ``close()`` is collective and idempotent; closed instances cannot
    be reused or consumed by an outstanding backward graph.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        num_experts: int,
        hidden_size: int,
        top_k: int,
        max_tokens_per_rank: int,
    ):
        if min(num_experts, hidden_size, top_k, max_tokens_per_rank) < 1:
            raise ValueError("DeepEP dimensions and token capacity must be positive.")
        if top_k > num_experts:
            raise ValueError("DeepEP top_k must not exceed num_experts.")
        self.group = group
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.top_k = int(top_k)
        self.max_tokens_per_rank = int(max_tokens_per_rank)
        self._buffer = None
        self._num_sms = 0
        self._index_dtype = torch.int64
        self._closed = False

    def _get_buffer(self):
        if self._closed:
            raise RuntimeError("DeepEP draft buffer is closed; finish backward before unloading.")
        if self._buffer is None:
            raise RuntimeError("DeepEP draft buffer has not been initialized.")
        return self._buffer

    def _ensure_buffer(self, device: torch.device) -> None:
        if self._closed:
            raise RuntimeError("A closed DeepEP draft dispatcher cannot be reused.")
        if self._buffer is not None:
            return
        if self.hidden_size % 256:
            raise ValueError("DeepEP V2 BF16 combine requires hidden_size divisible by 256.")
        ElasticBuffer, self._index_dtype = _load_elastic_buffer()
        if not dist.is_initialized() or dist.get_backend(self.group) != "nccl":
            raise RuntimeError("DeepEP requires an initialized NCCL expert process group.")
        ep_size = dist.get_world_size(self.group)
        if self.num_experts % ep_size:
            raise ValueError("The number of experts must be divisible by the DeepEP group size.")
        geometry = torch.tensor(
            [self.num_experts, self.hidden_size, self.top_k, self.max_tokens_per_rank],
            dtype=torch.int64,
            device=device,
        )
        rank_geometries = [torch.empty_like(geometry) for _ in range(ep_size)]
        dist.all_gather(rank_geometries, geometry, group=self.group)
        if any(not torch.equal(other, geometry) for other in rank_geometries):
            raise ValueError("DeepEP buffer geometry and token capacity must match on all EP ranks.")
        self._buffer = ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=self.max_tokens_per_rank,
            hidden=self.hidden_size,
            num_topk=self.top_k,
            use_fp8_dispatch=False,
            deterministic=torch.are_deterministic_algorithms_enabled(),
            explicitly_destroy=True,
        )
        try:
            self._num_sms = self._buffer.get_theoretical_num_sms(
                self.num_experts, self.top_k
            )
        except ZeroDivisionError:
            # Some RDMA topologies report zero link bandwidth. SM count only
            # affects performance; match the existing Titan conservative fallback.
            self._num_sms = min(20, torch.cuda.get_device_properties(device).multi_processor_count)

    def _validate_inputs(self, hidden, indices, scores) -> None:
        if torch.compiler.is_compiling():
            raise RuntimeError("The draft DeepEP backend currently requires eager execution.")
        if hidden.ndim != 2 or hidden.shape[1] != self.hidden_size:
            raise ValueError("DeepEP hidden states must have shape [tokens, hidden_size].")
        expected = (hidden.shape[0], self.top_k)
        if indices.shape != expected or scores.shape != expected:
            raise ValueError("DeepEP routing indices and scores must have shape [tokens, top_k].")
        if hidden.shape[0] > self.max_tokens_per_rank:
            raise ValueError("DeepEP token count exceeds expert_dispatch_max_tokens_per_rank.")
        if indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("DeepEP routing indices must use an integer dtype.")
        if not scores.is_floating_point():
            raise TypeError("DeepEP routing scores must use a floating-point dtype.")
        if hidden.dtype != torch.bfloat16 or hidden.device.type != "cuda":
            raise TypeError("The draft DeepEP training backend requires CUDA BF16 hidden states.")
        if indices.device != hidden.device or scores.device != hidden.device:
            raise ValueError("DeepEP hidden states and routing tensors must be on the same CUDA device.")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DeepEP compact training dispatch cannot run in a CUDA graph.")

    def __call__(
        self,
        hidden_states: torch.Tensor,
        global_indices: torch.Tensor,
        scores: torch.Tensor,
        expert_forward: Callable,
    ) -> torch.Tensor:
        self._validate_inputs(hidden_states, global_indices, scores)
        self._ensure_buffer(hidden_states.device)
        num_tokens = hidden_states.shape[0]
        if num_tokens == 0:
            # Still participate in all collectives when this rank has no input.
            # A disabled sentinel avoids relying on zero-sized CUDA launches.
            hidden_states = torch.cat([hidden_states, hidden_states.new_zeros((1, self.hidden_size))])
            global_indices = global_indices.new_full((1, self.top_k), -1)
            scores = torch.cat([scores, scores.new_zeros((1, self.top_k))])
        state = _DispatchState(self)
        recv_hidden, recv_indices, recv_scores = _Dispatch.apply(
            hidden_states.contiguous(),
            global_indices.to(self._index_dtype).contiguous(),
            scores.float().contiguous(),
            state,
        )
        # Compact dispatch deduplicates tokens per destination rank. Expand
        # every valid local expert selection before the existing grouped GEMM.
        # Keep zero-valued scores: their router derivatives need not be zero.
        token_ids, slots = torch.where(recv_indices >= 0)
        local_ids = recv_indices[token_ids, slots]
        order = torch.argsort(local_ids, stable=True)
        token_ids, slots, local_ids = token_ids[order], slots[order], local_ids[order]
        expert_input = recv_hidden.index_select(0, token_ids)
        expert_output = expert_forward(
            expert_input,
            local_ids.to(torch.long).unsqueeze(-1),
            expert_input.new_ones((expert_input.shape[0], 1)),
        )
        # Some expert implementations return a disconnected empty tensor when
        # no expert was selected. Keep dispatch backward in the collective order.
        expert_output = expert_output + expert_input * 0
        weighted = expert_output * recv_scores[token_ids, slots].to(expert_output.dtype).unsqueeze(-1)
        compact_output = recv_hidden.new_zeros(recv_hidden.shape).index_add(0, token_ids, weighted)
        return _Combine.apply(compact_output, state)[:num_tokens]

    def close(self) -> None:
        """Release the communication buffer collectively after all backward work."""
        if self._closed:
            return
        if self._buffer is not None:
            self._buffer.destroy()
            self._buffer = None
        self._closed = True


__all__ = ["DeepEPDispatcher", "require_deepep"]
