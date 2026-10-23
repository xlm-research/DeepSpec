"""Qwen3.5 supervision on selected complete assistant segments."""

from typing import Any

from torchspec.data.reasoning import REASONING_FORMAT_VERSION, apply_chat_template_with_reasoning
from torchspec.data.renderers.qwen3_5_multimodal import Qwen35MultimodalRenderer


class Qwen35FullDataRenderer(Qwen35MultimodalRenderer):
    CACHE_VERSION = (
        f"qwen35-full-data-native-system-selected-segments-thinking-v2-{REASONING_FORMAT_VERSION}"
    )

    def __init__(self, tokenizer: Any) -> None:
        super().__init__(tokenizer)
        template = tokenizer.get_chat_template()
        if "preserve_thinking" not in template:
            condition = "{%- if loop.index0 > ns.last_query_index %}"
            if template.count(condition) != 1:
                raise ValueError("Unsupported native template for historical reasoning")
            template = template.replace(
                condition, "{%- if preserve_thinking or loop.index0 > ns.last_query_index %}"
            )
        rejection = "{{- raise_exception('System message must be at the beginning.') }}"
        if template.count(rejection) != 1:
            raise ValueError("Unsupported native template for ordered later system messages")
        replacement = (
            "{{- '<|im_start|>system\\n' + render_content(message.content, false, true)|trim "
            "+ '<|im_end|>\\n' }}"
        )
        self.full_data_chat_template = template.replace(rejection, replacement)

    def format(self, messages: Any, tools: Any = None, *, generation_config: Any = None) -> str:
        options = dict(generation_config or {})
        options.pop("supervision_window", None)
        options.update(
            tools=tools,
            preserve_thinking=True,
            add_generation_prompt=False,
            tokenize=False,
            chat_template=self.full_data_chat_template,
        )
        return apply_chat_template_with_reasoning(self.tokenizer, messages, **options)

    def compute_loss_mask(
        self,
        input_ids: Any,
        *,
        last_turn_only: bool = False,
        supervision_window: dict[str, Any] | None = None,
    ) -> list[int]:
        mask = super().compute_loss_mask(input_ids, last_turn_only=last_turn_only)
        if supervision_window is None:
            return mask
        ids = input_ids.tolist() if hasattr(input_ids, "tolist") else list(input_ids)
        header, _, _ = self.get_assistant_token_ids()
        starts = [
            i
            for i, token in enumerate(ids)
            if token == header[0] and ids[i : i + len(header)] == header
        ]
        if len(starts) != supervision_window["assistant_count"]:
            raise ValueError("Rendered assistant boundaries disagree with the supervision window")
        ordinals = supervision_window["supervised_assistant_ordinals"]
        selected = set(ordinals)
        if (
            not selected
            or len(selected) != len(ordinals)
            or any(
                type(ordinal) is not int or not 0 <= ordinal < len(starts) for ordinal in selected
            )
        ):
            raise ValueError("Invalid selected assistant segments")
        for ordinal, start in enumerate(starts):
            if ordinal not in selected:
                end = starts[ordinal + 1] if ordinal + 1 < len(starts) else len(ids)
                mask[start:end] = [0] * (end - start)
        return mask

    def render(
        self,
        messages: Any,
        tools: Any = None,
        *,
        max_seq_length: int,
        last_turn_only: bool = False,
        generation_config: Any = None,
    ) -> tuple[list[int], list[int]]:
        ids, mask = super().render(
            messages,
            tools,
            max_seq_length=max_seq_length,
            last_turn_only=last_turn_only,
            generation_config=generation_config,
        )
        selection = (generation_config or {}).get("supervision_window")
        if selection is not None:
            mask = self.compute_loss_mask(ids, supervision_window=selection)
        return ids, mask
