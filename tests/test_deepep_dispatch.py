"""CPU protocol tests and an opt-in real DeepEP distributed numerical check.

Real hardware check (Hopper or newer, compatible draft environment):
    DEEPSPEC_TEST_DEEPEP=1 torchrun --nproc-per-node=2 -m unittest \
        tests.test_deepep_dispatch.DeepEPHardwareTest

On a single NVLink domain without NCCL GIN, add EP_DISABLE_GIN=1 to this
command. Cross-node EP requires its own working RDMA/GIN configuration.
"""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist

from deepspec.distributed.deepep_dispatch import (
    DeepEPDispatcher,
    _loaded_nccl_version,
    require_deepep,
)


class _Event:
    def __init__(self, log):
        self.log = log

    def current_stream_wait(self):
        self.log.append("wait")


class _FakeBuffer:
    """Emulate the V2 compact protocol on CPU; not a communication benchmark."""

    def __init__(self):
        self.log = []
        self.destroy_count = 0

    def dispatch(self, hidden, *, handle=None, topk_idx=None, topk_weights=None, **kwargs):
        self.log.append("dispatch_backward" if handle is not None else "dispatch_forward")
        if handle is None:
            assert kwargs["do_expand"] is False
            assert kwargs["do_cpu_sync"] is True
            rows = torch.where((topk_idx >= 0).any(dim=-1))[0]
            handle = SimpleNamespace(
                rows=rows,
                num_sms=kwargs["num_sms"],
                source_shape=hidden.shape,
                score_shape=topk_weights.shape,
                indices=topk_idx.index_select(0, rows),
                capacity=kwargs["num_max_tokens_per_rank"],
            )
            scores = topk_weights.index_select(0, rows)
            indices = handle.indices
        else:
            assert kwargs["do_cpu_sync"] is False
            assert kwargs["num_sms"] == handle.num_sms
            scores, indices = None, None
        return hidden.index_select(0, handle.rows), indices, scores, handle, _Event(self.log)

    def combine(self, hidden, *, handle, topk_weights, **kwargs):
        assert kwargs["async_with_compute_stream"] is True
        self.log.append("combine_backward" if topk_weights is not None else "combine_forward")
        combined = hidden.new_zeros(handle.source_shape).index_add(0, handle.rows, hidden)
        scores = None
        if topk_weights is not None:
            scores = topk_weights.new_zeros(handle.score_shape).index_add(
                0, handle.rows, topk_weights
            )
        return combined, scores, _Event(self.log)

    def destroy(self):
        self.destroy_count += 1


def _experts(weight, *, record=None):
    def forward(hidden, indices, unit_weights):
        if record is not None:
            record.append(indices.detach().clone())
        torch.testing.assert_close(unit_weights, torch.ones_like(unit_weights))
        expert_ids = indices.squeeze(-1)
        if expert_ids.numel() > 1:
            assert bool(torch.all(expert_ids[1:] >= expert_ids[:-1]))
        # Batched matrix multiplication leaves a zero gradient for unselected
        # experts and works with empty expert inputs.
        return torch.bmm(hidden.unsqueeze(1), weight[expert_ids]).squeeze(1)

    return forward


def _reference(hidden, indices, scores, weight):
    tokens, slots = torch.where(indices >= 0)
    values = torch.bmm(hidden[tokens].unsqueeze(1), weight[indices[tokens, slots]]).squeeze(1)
    weighted = values * scores[tokens, slots].to(values.dtype).unsqueeze(-1)
    return hidden.new_zeros(hidden.shape).index_add(0, tokens, weighted)


class DeepEPDispatcherTest(unittest.TestCase):
    def _dispatcher(self):
        dispatcher = DeepEPDispatcher(
            object(), num_experts=4, hidden_size=8, top_k=2, max_tokens_per_rank=16
        )
        dispatcher._buffer = _FakeBuffer()
        dispatcher._num_sms = 4
        # Production validation requires CUDA BF16. The protocol itself can be
        # tested on CPU without importing DeepEP or launching GPU work.
        dispatcher._validate_inputs = mock.Mock()
        return dispatcher

    def test_forward_and_hidden_router_expert_gradients(self):
        torch.manual_seed(413)
        dispatcher = self._dispatcher()
        hidden = torch.randn(5, 8, dtype=torch.bfloat16, requires_grad=True)
        scores = torch.tensor(
            [[0.25, 0.75], [0.0, 0.5], [0.2, 0.8], [0.5, 0.5], [1.0, 0.0]],
            requires_grad=True,
        )
        # Multiple local experts per token, repeated picks, an unused expert,
        # a disabled slot, and exact-zero scores all exercise compact mapping.
        indices = torch.tensor([[2, 0], [1, 0], [2, 2], [1, -1], [0, 1]])
        weight = torch.randn(4, 8, 8, dtype=torch.bfloat16, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_()
        reference_scores = scores.detach().clone().requires_grad_()
        reference_weight = weight.detach().clone().requires_grad_()
        order = []
        actual = dispatcher(hidden, indices, scores, _experts(weight, record=order))
        expected = _reference(reference_hidden, indices, reference_scores, reference_weight)
        torch.testing.assert_close(actual, expected, atol=0.04, rtol=0.02)
        grad = torch.randn_like(actual)
        actual.backward(grad)
        expected.backward(grad)
        for actual_grad, expected_grad in (
            (hidden.grad, reference_hidden.grad),
            (scores.grad, reference_scores.grad),
            (weight.grad, reference_weight.grad),
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=0.06, rtol=0.04)
        self.assertEqual(scores.grad.dtype, torch.float32)
        self.assertNotEqual(float(scores.grad[1, 0]), 0.0)
        self.assertEqual(int(torch.count_nonzero(weight.grad[3])), 0)
        self.assertEqual(
            dispatcher._buffer.log,
            ["dispatch_forward", "wait", "combine_forward", "wait",
             "dispatch_backward", "wait", "combine_backward", "wait"],
        )

    def test_empty_tokens_and_no_selected_experts_still_run_backward(self):
        for num_tokens in (0, 3):
            with self.subTest(num_tokens=num_tokens):
                dispatcher = self._dispatcher()
                hidden = torch.zeros(num_tokens, 8, dtype=torch.bfloat16, requires_grad=True)
                scores = torch.zeros(num_tokens, 2, requires_grad=True)
                indices = torch.full((num_tokens, 2), -1, dtype=torch.int64)
                weight = torch.ones(4, 8, 8, dtype=torch.bfloat16, requires_grad=True)
                actual = dispatcher(hidden, indices, scores, _experts(weight))
                self.assertEqual(actual.shape, hidden.shape)
                actual.sum().backward()
                torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden))
                torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
                self.assertIn("combine_backward", dispatcher._buffer.log)

    def test_shared_buffer_retains_distinct_layer_handles(self):
        dispatcher = self._dispatcher()
        weight = torch.ones(4, 8, 8, dtype=torch.bfloat16, requires_grad=True)
        hidden_a = torch.ones(2, 8, dtype=torch.bfloat16, requires_grad=True)
        hidden_b = torch.ones(5, 8, dtype=torch.bfloat16, requires_grad=True)
        scores_a = torch.full((2, 2), 0.5, requires_grad=True)
        scores_b = torch.full((5, 2), 0.5, requires_grad=True)
        out_a = dispatcher(hidden_a, torch.tensor([[0, 1], [1, 2]]), scores_a, _experts(weight))
        out_b = dispatcher(hidden_b, torch.zeros(5, 2, dtype=torch.long), scores_b, _experts(weight))
        (out_a.sum() + out_b.sum()).backward()
        torch.testing.assert_close(hidden_a.grad, torch.full_like(hidden_a, 8))
        torch.testing.assert_close(hidden_b.grad, torch.full_like(hidden_b, 8))

    def test_close_is_idempotent_and_rejects_later_use(self):
        dispatcher = self._dispatcher()
        buffer = dispatcher._buffer
        dispatcher.close()
        dispatcher.close()
        self.assertEqual(buffer.destroy_count, 1)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            dispatcher._get_buffer()
        with self.assertRaisesRegex(RuntimeError, "closed"):
            dispatcher._ensure_buffer(torch.device("cpu"))

    def test_input_validation_and_preflight(self):
        dispatcher = DeepEPDispatcher(
            object(), num_experts=4, hidden_size=8, top_k=2, max_tokens_per_rank=4
        )
        with self.assertRaisesRegex(ValueError, "token count"):
            dispatcher._validate_inputs(torch.zeros(5, 8), torch.zeros(5, 2), torch.zeros(5, 2))
        with self.assertRaisesRegex(TypeError, "CUDA BF16"):
            dispatcher._validate_inputs(
                torch.zeros(2, 8), torch.zeros(2, 2, dtype=torch.long), torch.zeros(2, 2)
            )
        with self.assertRaisesRegex(ValueError, "divisible by 256"):
            dispatcher._ensure_buffer(torch.device("cpu"))
        with mock.patch.object(dist, "is_nccl_available", return_value=True), mock.patch(
            "deepspec.distributed.deepep_dispatch._loaded_nccl_version", return_value=(2, 28, 9)
        ):
            with self.assertRaisesRegex(RuntimeError, "NCCL >= 2.30.4"):
                require_deepep()

    def test_preflight_queries_loaded_nccl_without_loading_another_library(self):
        path = "/draft/site-packages/nvidia/nccl/lib/libnccl.so.2"
        maps = f"1000-2000 r--p 00000000 01:01 123 {path}\n2000-3000 r-xp 00000000 01:01 123 {path}\n"

        def get_version(pointer):
            pointer._obj.value = 23004
            return 0

        library = SimpleNamespace(ncclGetVersion=mock.Mock(side_effect=get_version))
        with mock.patch("builtins.open", mock.mock_open(read_data=maps)), mock.patch(
            "deepspec.distributed.deepep_dispatch.ctypes.CDLL", return_value=library
        ) as load_library:
            self.assertEqual(_loaded_nccl_version(), (2, 30, 4))
        load_library.assert_called_once_with(
            path, mode=os.RTLD_NOLOAD | os.RTLD_NOW | os.RTLD_LOCAL
        )
        with mock.patch("builtins.open", mock.mock_open(read_data=maps + maps.replace(path, "/other/libnccl.so.2"))):
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                _loaded_nccl_version()
        with mock.patch.object(dist, "is_nccl_available", return_value=True), mock.patch(
            "deepspec.distributed.deepep_dispatch._loaded_nccl_version", return_value=(2, 30, 4)
        ), mock.patch("deepspec.distributed.deepep_dispatch._load_elastic_buffer"), mock.patch(
            "torch.cuda.nccl.version", side_effect=AssertionError("build-time version must not be used")
        ):
            require_deepep()


@unittest.skipUnless(os.environ.get("DEEPSPEC_TEST_DEEPEP") == "1", "opt-in DeepEP GPU test")
class DeepEPHardwareTest(unittest.TestCase):
    def test_uneven_tokens_and_global_expert_gradients(self):
        if not dist.is_initialized():
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            dist.init_process_group("nccl")
        require_deepep()
        rank, world = dist.get_rank(), dist.get_world_size()
        device = torch.device("cuda", torch.cuda.current_device())
        num_experts, hidden_size, top_k = 4 * world, 512, 2
        dispatcher = DeepEPDispatcher(
            dist.group.WORLD,
            num_experts=num_experts,
            hidden_size=hidden_size,
            top_k=top_k,
            max_tokens_per_rank=32,
        )
        try:
            for case in ("uneven", "one_empty", "all_empty", "skewed"):
                torch.manual_seed(61)
                all_weight = torch.randn(num_experts, hidden_size, hidden_size, device=device, dtype=torch.bfloat16) / 16
                local_weight = all_weight.chunk(world)[rank].clone().requires_grad_()
                reference_weight = all_weight.clone().requires_grad_()
                num_tokens = (
                    0 if case == "all_empty" or (case == "one_empty" and rank == 0)
                    else 3 + rank
                )
                torch.manual_seed(100 + rank)
                hidden = torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
                # In the skewed case, every token selects rank 0's experts;
                # all other ranks must participate with zero received tokens.
                routed_experts = num_experts // world if case == "skewed" else num_experts
                indices = torch.arange(num_tokens * top_k, device=device).reshape(num_tokens, top_k) % routed_experts
                scores = torch.rand(num_tokens, top_k, device=device, requires_grad=True)
                reference_hidden = hidden.detach().clone().requires_grad_()
                reference_scores = scores.detach().clone().requires_grad_()
                actual = dispatcher(hidden, indices, scores, _experts(local_weight))
                expected = _reference(reference_hidden, indices, reference_scores, reference_weight)
                torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)
                actual.sum().backward()
                expected.sum().backward()
                dist.all_reduce(reference_weight.grad)
                for actual_grad, expected_grad in (
                    (hidden.grad, reference_hidden.grad),
                    (scores.grad, reference_scores.grad),
                    (local_weight.grad, reference_weight.grad.chunk(world)[rank]),
                ):
                    torch.testing.assert_close(actual_grad, expected_grad, rtol=0.06, atol=0.08)
        finally:
            dispatcher.close()


if __name__ == "__main__":
    unittest.main()
