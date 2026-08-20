import types
import unittest

import torch
from torch import nn

from deepspec.data.parser import (
    MultimodalTruncationError,
    normalize_multimodal_messages,
    preprocess_multimodal_record,
)
from deepspec.data.target_cache_dataset import MultimodalConversationCollator
from scripts.data.prepare_target_cache import run_target_forward_with_hooks


class _FakeTokenizer:
    padding_side = "left"
    pad_token_id = 0


class _FakeMediaProcessor:
    merge_size = 2


class _FakeProcessor:
    tokenizer = _FakeTokenizer()
    image_processor = _FakeMediaProcessor()
    video_processor = _FakeMediaProcessor()
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
        video_grids = []

        def append(token, token_type=0):
            input_ids.append(token)
            token_types.append(token_type)

        for message in messages:
            append({"system": 10, "user": 20, "assistant": 30, "tool": 40}[message["role"]])
            content = message["content"]
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            for block in blocks:
                if block["type"] == "image":
                    append(91, 1)
                    for _ in range(4):
                        append(self.image_token_id, 1)
                    append(92, 1)
                    image_grids.append([1, 4, 4])
                elif block["type"] == "video":
                    append(93, 2)
                    for _ in range(4):
                        append(self.video_token_id, 2)
                    append(94, 2)
                    video_grids.append([1, 4, 4])
                else:
                    for _word in str(block.get("text", "")).split():
                        append(50)
            append(2)
        if add_generation_prompt:
            append(30)

        result = {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(input_ids)), dtype=torch.long),
            "mm_token_type_ids": torch.tensor([token_types], dtype=torch.long),
        }
        if image_grids:
            result["image_grid_thw"] = torch.tensor(image_grids, dtype=torch.long)
            result["pixel_values"] = torch.ones((16 * len(image_grids), 3))
        if video_grids:
            result["video_grid_thw"] = torch.tensor(video_grids, dtype=torch.long)
            result["pixel_values_videos"] = torch.ones((16 * len(video_grids), 3))
        return result


class MultimodalParserTest(unittest.TestCase):
    def setUp(self):
        self.processor = _FakeProcessor()
        self.record = {
            "messages": [
                {"from": "human", "value": "<image> describe this"},
                {"from": "gpt", "value": "a white square"},
            ],
            "images": ["relative.png"],
        }

    def test_normalizes_ms_swift_record(self):
        messages = normalize_multimodal_messages(
            self.record,
            chat_template="qwen",
            media_root="/data/images",
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"][0]["type"], "image")
        self.assertEqual(
            messages[1]["content"][0]["image"],
            "/data/images/relative.png",
        )

    def test_remaps_media_uri_using_longest_prefix(self):
        record = {
            "messages": [
                {"role": "user", "content": "<image> describe this"},
                {"role": "assistant", "content": "a white square"},
            ],
            "images": ["s3://bucket/special/nested/image.png"],
        }
        messages = normalize_multimodal_messages(
            record,
            chat_template="qwen",
            media_uri_map={
                "s3://bucket/": "/general/",
                "s3://bucket/special/": "/local/materials/",
            },
        )
        self.assertEqual(
            messages[1]["content"][0]["image"],
            "/local/materials/nested/image.png",
        )

    def test_processor_outputs_and_assistant_mask_are_preserved(self):
        processed = preprocess_multimodal_record(
            self.record,
            processor=self.processor,
            chat_template="qwen",
            max_length=64,
            media_root="/data/images",
        )
        self.assertIn("pixel_values", processed)
        self.assertIn("image_grid_thw", processed)
        self.assertIn("mm_token_type_ids", processed)
        self.assertEqual(int(processed["loss_mask"].sum()), 4)
        self.assertEqual(
            int((processed["input_ids"] == self.processor.image_token_id).sum()),
            4,
        )

    def test_rejects_truncation_through_visual_tokens(self):
        with self.assertRaises(MultimodalTruncationError):
            preprocess_multimodal_record(
                self.record,
                processor=self.processor,
                chat_template="qwen",
                max_length=5,
                media_root="/data/images",
            )

    def test_collator_left_pads_and_concatenates_media(self):
        longer = {
            "messages": [
                {"role": "user", "content": "<image> describe this now"},
                {"role": "assistant", "content": "a white square"},
            ],
            "images": ["second.png"],
        }
        collator = MultimodalConversationCollator(
            processor=self.processor,
            chat_template="qwen",
            max_length=64,
            min_loss_tokens=1,
            media_root="/data/images",
        )
        batch = collator([self.record, longer])
        self.assertEqual(batch["input_ids"].shape[0], 2)
        self.assertEqual(batch["input_ids"][0, 0].item(), 0)
        self.assertEqual(batch["pixel_values"].shape[0], 32)
        self.assertEqual(batch["image_grid_thw"].shape, (2, 3))


class _FakeLayer(nn.Module):
    def forward(self, hidden_states, **_kwargs):
        return hidden_states + 1


class _FakeTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList([_FakeLayer(), _FakeLayer()])
        self.language_model.norm = nn.Identity()
        self.saw_pixel_values = False

    def forward(self, input_ids, pixel_values=None, **_kwargs):
        self.saw_pixel_values = pixel_values is not None
        hidden_states = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        for layer in self.language_model.layers:
            hidden_states = layer(hidden_states)
        hidden_states = self.language_model.norm(hidden_states)
        return types.SimpleNamespace(last_hidden_state=hidden_states)


class TargetForwardTest(unittest.TestCase):
    def test_multimodal_inputs_reach_target_and_decoder_hooks(self):
        model = _FakeTarget()
        result = run_target_forward_with_hooks(
            target_model=model,
            model_inputs={
                "input_ids": torch.tensor([[1, 2, 3]]),
                "pixel_values": torch.ones((4, 3)),
            },
            target_layer_ids=[-1, 1],
        )
        self.assertTrue(model.saw_pixel_values)
        self.assertEqual(result.target_hidden_states.shape, (1, 3, 6))
        self.assertEqual(result.target_last_hidden_states.shape, (1, 3, 3))


if __name__ == "__main__":
    unittest.main()
