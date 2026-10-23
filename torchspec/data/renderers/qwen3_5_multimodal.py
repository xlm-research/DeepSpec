"""Native Qwen3.5 conversations with supervision on all assistant bodies."""

import torch

from torchspec.data.reasoning import REASONING_FORMAT_VERSION, apply_chat_template_with_reasoning
from torchspec.models.ops.loss_mask import compute_assistant_loss_mask


class Qwen35MultimodalRenderer:
    CACHE_VERSION = f"qwen35-native-vision-mask-v1-{REASONING_FORMAT_VERSION}"
    RECOMPUTE_PREFILL_MASK = True

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.visual_token_ids = {
            tokenizer.convert_tokens_to_ids(token)
            for token in ("<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>")
        }

    def get_assistant_token_ids(self):
        header = "<|im_start|>assistant"
        ids = self.tokenizer.encode(header, add_special_tokens=False)
        full = self.tokenizer.encode(header + "\n", add_special_tokens=False)
        return (
            ids,
            self.tokenizer.encode("<|im_end|>", add_special_tokens=False),
            len(full) - len(ids),
        )

    def format(self, messages, tools=None, *, generation_config=None):
        options = dict(generation_config or {})
        options.update(
            tools=tools, preserve_thinking=True, add_generation_prompt=False, tokenize=False
        )
        return apply_chat_template_with_reasoning(self.tokenizer, messages, **options)

    def compute_loss_mask(self, input_ids, *, last_turn_only=False):
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        header, end, skip = self.get_assistant_token_ids()
        mask = compute_assistant_loss_mask(
            ids, header, end, last_turn_only=last_turn_only, skip_after_header=skip
        )
        for token in self.visual_token_ids:
            mask[ids == token] = 0
        return mask.tolist()

    def render(
        self, messages, tools=None, *, max_seq_length, last_turn_only=False, generation_config=None
    ):
        text = self.format(messages, tools, generation_config=generation_config)
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        # Truncating an image conversation can separate its placeholder from its payload.
        if len(ids) > max_seq_length:
            raise ValueError(
                f"Native image conversation has {len(ids)} tokens; limit is {max_seq_length}"
            )
        return ids, self.compute_loss_mask(ids, last_turn_only=last_turn_only)
