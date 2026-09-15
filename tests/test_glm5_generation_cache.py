"""Check bounded GLM prefill can seed correct token-by-token decoding."""

import copy
import unittest
from unittest.mock import patch

import torch
from transformers import DynamicCache
from transformers.models.glm5_next.modeling_glm5_next import (
    Glm5NextTextModel,
    chunk_kimi_delta_attention,
)

from deepspec.modeling.target.glm5_next import install_glm5_next_bounded_target_prefill
from tests.test_glm5_next_dspark import tiny_target_config


@unittest.skipUnless(torch.cuda.is_available(), "CUDA causal-conv decode is required")
class Glm5GenerationCacheTest(unittest.TestCase):
    def test_cached_decode_matches_full_recomputation(self):
        torch.manual_seed(19)
        torch.set_float32_matmul_precision("highest")
        config = tiny_target_config().text_config
        config.linear_num_heads = 4
        config.linear_head_dim = 8
        config.linear_conv_kernel_dim = 4
        config.linear_lower_bound = -5.0
        config._attn_implementation = "eager"
        original = Glm5NextTextModel(config).cuda().eval()
        bounded = copy.deepcopy(original)
        install_glm5_next_bounded_target_prefill(bounded)
        ids = torch.randint(0, 120, (1, 78), device="cuda")
        cache = DynamicCache(config=config)

        # Avoid unrelated compilation in the short reference forward. The
        # recurrence itself is unchanged; decode still uses the native path.
        reference_chunk = getattr(
            chunk_kimi_delta_attention, "__wrapped__", chunk_kimi_delta_attention
        )
        with (
            torch.no_grad(),
            patch(
                "transformers.models.glm5_next.modeling_glm5_next.chunk_kimi_delta_attention",
                reference_chunk,
            ),
        ):
            cached = bounded(
                input_ids=ids[:, :73], use_cache=True, past_key_values=cache
            )
            expected = original(input_ids=ids[:, :73], use_cache=False)
            uncached = bounded(input_ids=ids[:, :73], use_cache=False)
            torch.testing.assert_close(
                cached.last_hidden_state,
                expected.last_hidden_state,
                atol=3e-4,
                rtol=3e-4,
            )
            torch.testing.assert_close(
                cached.last_hidden_state,
                uncached.last_hidden_state,
                atol=1e-6,
                rtol=1e-6,
            )
            for position in range(73, 78):
                cached = bounded(
                    input_ids=ids[:, position : position + 1],
                    use_cache=True,
                    past_key_values=cache,
                )
                expected = original(input_ids=ids[:, : position + 1], use_cache=False)
                torch.testing.assert_close(
                    cached.last_hidden_state[:, -1],
                    expected.last_hidden_state[:, -1],
                    atol=3e-4,
                    rtol=3e-4,
                )
                self.assertEqual(cache.get_seq_length(), position + 1)
            for index, kind in enumerate(config.layer_types):
                if kind == "linear_attention":
                    self.assertTrue(cache.has_previous_state(index))
                    self.assertTrue(
                        torch.isfinite(cache.layers[index].recurrent_states[0]).all()
                    )
                else:
                    self.assertEqual(cache.layers[index].keys.shape[-2], 78)
                    self.assertEqual(cache.layers[index].indexer_keys.shape[1], 78)


if __name__ == "__main__":
    unittest.main()
