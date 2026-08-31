import os
import types
import unittest

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from deepspec.data.parser import (
    encode_multimodal_generation_record,
    preprocess_multimodal_record,
)
from deepspec.modeling.dspark.qwen3_6.config import build_draft_config
from deepspec.modeling.dspark.qwen3_6.modeling import Qwen3_6DSparkModel
from deepspec.modeling.target_adapter import (
    Qwen3_6TargetAdapter,
    get_target_adapter,
)
from deepspec.utils.config import ConfigNode


class _FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0


class _FakeImageProcessor:
    merge_size = 2


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeImageProcessor()
    video_processor = _FakeImageProcessor()
    image_token_id = 99
    video_token_id = 98

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
        return_tensors,
        processor_kwargs,
    ):
        del tokenize, return_dict, return_tensors, processor_kwargs
        input_ids = []
        token_types = []
        image_grids = []
        for message in messages:
            input_ids.append(
                {"system": 10, "user": 20, "assistant": 30, "tool": 40}[
                    message["role"]
                ]
            )
            token_types.append(0)
            content = message["content"]
            blocks = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": content}]
            )
            for block in blocks:
                if block["type"] == "image":
                    input_ids.extend([91, 99, 99, 99, 99, 92])
                    token_types.extend([1] * 6)
                    image_grids.append([1, 4, 4])
                else:
                    words = str(block.get("text", "")).split()
                    input_ids.extend([50] * len(words))
                    token_types.extend([0] * len(words))
            input_ids.append(2)
            token_types.append(0)
        if add_generation_prompt:
            input_ids.append(30)
            token_types.append(0)
        result = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([token_types], dtype=torch.long),
        }
        if image_grids:
            result["image_grid_thw"] = torch.tensor(image_grids, dtype=torch.long)
            result["pixel_values"] = torch.ones((16 * len(image_grids), 3))
        return result


class Qwen3_6AdapterTest(unittest.TestCase):
    def test_qwen35_architecture_selects_qwen36_adapter(self):
        config = types.SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
        )
        adapter = get_target_adapter(config, "/models/Qwen3.6-27B")
        self.assertIsInstance(adapter, Qwen3_6TargetAdapter)

    def test_generation_encoder_strips_reference_answer(self):
        processor = _FakeProcessor()
        record = {
            "messages": [
                {"role": "user", "content": "<image> identify the color"},
                {"role": "assistant", "content": "red"},
            ],
            "images": ["red.ppm"],
        }
        encoded = encode_multimodal_generation_record(
            record,
            processor=processor,
            chat_template="qwen",
            media_root="/tmp/media",
        )
        encoded_with_answer = encode_multimodal_generation_record(
            record,
            processor=processor,
            chat_template="qwen",
            media_root="/tmp/media",
            strip_assistant=False,
        )
        # System + user + generation assistant header, but no answer tokens.
        self.assertEqual(encoded["input_ids"][0, -1].item(), 30)
        self.assertEqual(
            int((encoded_with_answer["input_ids"] == 50).sum().item()),
            int((encoded["input_ids"] == 50).sum().item()) + 1,
        )
        self.assertIn("pixel_values", encoded)

    def test_text_prefill_clears_rope_deltas_from_previous_sample(self):
        adapter = Qwen3_6TargetAdapter()
        backbone = types.SimpleNamespace(rope_deltas=torch.tensor([[17]]))
        target = types.SimpleNamespace(model=backbone)
        cache = object()

        kwargs = adapter.build_prefill_inputs(
            target_model=target,
            model_inputs={"input_ids": torch.tensor([[1, 2, 3]])},
            cache=cache,
        )

        self.assertIsNone(backbone.rope_deltas)
        self.assertIs(kwargs["past_key_values"], cache)
        self.assertNotIn("position_ids", kwargs)

    def test_partial_rejection_restores_and_replays_hybrid_cache(self):
        adapter = Qwen3_6TargetAdapter()

        class _LinearLayer:
            def __init__(self):
                self.conv_states = {0: torch.tensor([1.0])}
                self.recurrent_states = {0: torch.tensor([2.0])}
                self.has_previous_state = {0: True}
                self.is_conv_states_initialized = {0: True}
                self.is_recurrent_states_initialized = {0: True}

        class _AttentionLayer:
            def __init__(self, length):
                self.length = length

            def get_seq_length(self):
                return self.length

            def crop(self, tokens_to_remove):
                self.length -= abs(int(tokens_to_remove))

        class _HybridCache:
            def __init__(self):
                self.layers = [_LinearLayer(), _AttentionLayer(4)]

            def get_seq_length(self):
                return self.layers[1].get_seq_length()

        class _Target:
            def __init__(self):
                self.calls = []

            def __call__(self, **kwargs):
                cache = kwargs["past_key_values"]
                self.calls.append({
                    "input_ids": kwargs["input_ids"].clone(),
                    "conv_state_before_replay": cache.layers[0].conv_states[0].clone(),
                    "recurrent_state_before_replay": (
                        cache.layers[0].recurrent_states[0].clone()
                    ),
                })
                cache.layers[1].length += kwargs["input_ids"].shape[1]
                return types.SimpleNamespace()

        cache = _HybridCache()
        adapter.build_verify_inputs(
            verify_input_ids=torch.tensor([[5, 6, 7, 8]]),
            position_ids=torch.arange(16).unsqueeze(0),
            start=4,
            cache=cache,
        )
        # Simulate the target verification mutating all hybrid cache layers.
        cache.layers[0].conv_states[0].fill_(11.0)
        cache.layers[0].recurrent_states[0].fill_(12.0)
        cache.layers[1].length = 8

        target = _Target()
        output_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 42, 0]])
        result = adapter.reconcile_generation_cache(
            target_model=target,
            cache=cache,
            model_inputs={"input_ids": output_ids[:, :4]},
            output_ids=output_ids,
            committed_length=6,
            accepted_draft_tokens=1,
            draft_token_count=3,
        )

        self.assertIs(result, cache)
        self.assertEqual(cache.get_seq_length(), 6)
        self.assertEqual(len(target.calls), 1)
        self.assertEqual(target.calls[0]["input_ids"].tolist(), [[5, 6]])
        torch.testing.assert_close(
            target.calls[0]["conv_state_before_replay"], torch.tensor([1.0])
        )
        torch.testing.assert_close(
            target.calls[0]["recurrent_state_before_replay"], torch.tensor([2.0])
        )
        self.assertFalse(hasattr(cache, adapter._CACHE_SNAPSHOT_ATTR))

    def test_partial_rejection_rebuilds_hybrid_cache(self):
        adapter = Qwen3_6TargetAdapter()
        old_cache = object()
        rebuilt_cache = object()
        adapter.create_generation_cache = lambda _model: rebuilt_cache

        class _Target:
            config = types.SimpleNamespace(model_type="qwen3_5")

            def __init__(self):
                self.kwargs = None

            def __call__(self, **kwargs):
                self.kwargs = kwargs
                return types.SimpleNamespace()

        target = _Target()
        model_inputs = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([[0, 1, 0]]),
            "pixel_values": torch.ones((4, 3)),
        }
        output_ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        result = adapter.reconcile_generation_cache(
            target_model=target,
            cache=old_cache,
            model_inputs=model_inputs,
            output_ids=output_ids,
            committed_length=5,
            accepted_draft_tokens=1,
            draft_token_count=3,
        )
        self.assertIs(result, rebuilt_cache)
        self.assertTrue(torch.equal(target.kwargs["input_ids"], output_ids[:, :5]))
        self.assertEqual(target.kwargs["mm_token_type_ids"].tolist(), [[0, 1, 0, 0, 0]])
        self.assertIs(target.kwargs["past_key_values"], rebuilt_cache)
        self.assertFalse(target.kwargs["output_hidden_states"])

    def test_all_accepted_keeps_hybrid_cache(self):
        adapter = Qwen3_6TargetAdapter()
        cache = object()
        result = adapter.reconcile_generation_cache(
            target_model=object(),
            cache=cache,
            model_inputs={"input_ids": torch.tensor([[1]])},
            output_ids=torch.tensor([[1, 2]]),
            committed_length=1,
            accepted_draft_tokens=3,
            draft_token_count=3,
        )
        self.assertIs(result, cache)

    def test_qwen36_draft_config_has_dedicated_architecture(self):
        text_config = types.SimpleNamespace(
            num_hidden_layers=64,
            hidden_size=32,
            vocab_size=128,
        )
        target_config = types.SimpleNamespace(text_config=text_config)
        model_args = ConfigNode({
            "num_draft_layers": 2,
            "target_layer_ids": [1, 31, 61],
            "confidence_head_alpha": 0.0,
            "markov_rank": 0,
            "block_size": 3,
            "mask_token_id": 127,
            "num_anchors": 8,
        })
        config = build_draft_config(target_config, model_args)
        self.assertEqual(config.architectures, ["Qwen3_6DSparkModel"])
        self.assertEqual(config.deepspec_target_family, "qwen3_6")
        self.assertEqual(config.target_layer_ids, [1, 31, 61])

    def test_tiny_qwen36_draft_model_builds(self):
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            layer_types=["full_attention"],
        )
        config.target_layer_ids = [1, 3]
        config.num_target_layers = 4
        config.block_size = 3
        config.mask_token_id = 127
        config.num_anchors = 4
        config.enable_confidence_head = False
        config.markov_rank = 0
        config._attn_implementation = "eager"
        model = Qwen3_6DSparkModel(config)
        self.assertEqual(model.embed_tokens.weight.shape, (128, 32))
        self.assertEqual(model.fc.in_features, 64)

    @unittest.skipUnless(torch.cuda.is_available(), "Qwen3.5 DeltaNet requires CUDA")
    def test_replayed_hybrid_cache_matches_fresh_forward(self):
        torch.manual_seed(0)
        config = Qwen3_5TextConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            max_position_embeddings=128,
            layer_types=["linear_attention"] * 3 + ["full_attention"],
            linear_key_head_dim=8,
            linear_value_head_dim=8,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.25,
            },
        )
        target = Qwen3_5ForCausalLM(config).eval().cuda()
        adapter = Qwen3_6TargetAdapter()
        prompt = torch.tensor([[1, 2, 3, 4]], device="cuda")
        verify_input_ids = torch.tensor([[5, 6, 7, 8]], device="cuda")
        output_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 42]], device="cuda")

        with torch.inference_mode():
            optimized_cache = adapter.create_generation_cache(target)
            target(
                input_ids=prompt,
                past_key_values=optimized_cache,
                use_cache=True,
            )
            target(**adapter.build_verify_inputs(
                verify_input_ids=verify_input_ids,
                position_ids=torch.arange(16, device="cuda").unsqueeze(0),
                start=4,
                cache=optimized_cache,
            ))
            result = adapter.reconcile_generation_cache(
                target_model=target,
                cache=optimized_cache,
                model_inputs={"input_ids": prompt},
                output_ids=output_ids,
                committed_length=6,
                accepted_draft_tokens=1,
                draft_token_count=3,
            )
            optimized_logits = target(
                input_ids=output_ids[:, 6:7],
                past_key_values=result,
                use_cache=True,
            ).logits

            fresh_cache = adapter.create_generation_cache(target)
            target(
                input_ids=output_ids[:, :6],
                past_key_values=fresh_cache,
                use_cache=True,
            )
            fresh_logits = target(
                input_ids=output_ids[:, 6:7],
                past_key_values=fresh_cache,
                use_cache=True,
            ).logits

        self.assertIs(result, optimized_cache)
        self.assertEqual(result.get_seq_length(), fresh_cache.get_seq_length())
        torch.testing.assert_close(
            optimized_logits,
            fresh_logits,
            atol=2e-5,
            rtol=2e-5,
        )


@unittest.skipUnless(
    os.environ.get("DEEPSPEC_QWEN36_MODEL_PATH"),
    "Set DEEPSPEC_QWEN36_MODEL_PATH to run the real processor integration test.",
)
class Qwen3_6RealProcessorTest(unittest.TestCase):
    def test_real_processor_encodes_smoke_image(self):
        model_path = os.environ["DEEPSPEC_QWEN36_MODEL_PATH"]
        repo_root = os.path.dirname(os.path.dirname(__file__))
        data_path = os.path.join(
            repo_root,
            "smoke_data",
            "qwen36_multimodal_smoke.jsonl",
        )
        import json

        with open(data_path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        record = records[0]
        processor = AutoProcessor.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)
        self.assertEqual(config.model_type, "qwen3_5")
        cache = Qwen3_6TargetAdapter().create_generation_cache(
            types.SimpleNamespace(config=config)
        )
        self.assertEqual(len(cache.layers), config.text_config.num_hidden_layers)
        self.assertTrue(
            any(type(layer).__name__ == "LinearAttentionLayer" for layer in cache.layers)
        )
        encoded = encode_multimodal_generation_record(
            record,
            processor=processor,
            chat_template="qwen",
            media_root=os.path.join(repo_root, "smoke_data"),
        )
        self.assertEqual(encoded["input_ids"].shape[0], 1)
        self.assertIn("pixel_values", encoded)
        self.assertIn("image_grid_thw", encoded)
        for training_record in records:
            processed = preprocess_multimodal_record(
                training_record,
                processor=processor,
                chat_template="qwen",
                max_length=256,
                media_root=os.path.join(repo_root, "smoke_data"),
            )
            self.assertGreater(int(processed["loss_mask"].sum().item()), 0)
            self.assertEqual(
                processed["input_ids"].shape[0],
                processed["attention_mask"].shape[0],
            )


if __name__ == "__main__":
    unittest.main()
