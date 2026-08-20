import torch

from deepspec.data.parser import GeneralParser, TEMPLATE_REGISTRY


class _Encoding:
    def __init__(self, text: str):
        self.input_ids = torch.tensor([[ord(ch) for ch in text]], dtype=torch.long)
        self.attention_mask = torch.ones_like(self.input_ids)


class _CharTokenizer:
    bos_token = "<｜begin▁of▁sentence｜>"
    eos_token = "<｜end▁of▁sentence｜>"

    def __init__(self):
        self.last_text = None

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, chat_template=None, **kwargs):
        assert tokenize is False
        assert add_generation_prompt is False
        parts = []
        for idx, message in enumerate(messages):
            if idx == 0:
                parts.append(self.bos_token)
            if message["role"] == "system":
                parts.append(message["content"])
            elif message["role"] == "user":
                parts.append("<｜User｜>" + message["content"])
            elif message["role"] == "assistant":
                parts.append("<｜Assistant｜>" + message["content"] + self.eos_token)
            else:
                raise AssertionError(message["role"])
        text = "".join(parts)
        self.last_text = text
        return text

    def __call__(self, text, *, max_length=None, truncation=False, return_tensors=None, add_special_tokens=False):
        if max_length is not None and truncation:
            text = text[:max_length]
        return _Encoding(text)

    def encode(self, text, *, add_special_tokens=False, truncation=False, max_length=None):
        if max_length is not None and truncation:
            text = text[:max_length]
        return [ord(ch) for ch in text]


def test_deepseek_v4_flash_body_does_not_inject_extra_think_close():
    tokenizer = _CharTokenizer()
    parser = GeneralParser(
        tokenizer=tokenizer,
        chat_template=TEMPLATE_REGISTRY.get("deepseek_v4_flash_body"),
    )
    assistant_body = (
        "<think>reasoning</think>answer\n\n"
        "<｜DSML｜tool_calls>\n</｜DSML｜tool_calls>"
    )

    out = parser.parse(
        [
            {"role": "user", "content": "do it"},
            {"role": "assistant", "content": assistant_body},
        ],
        max_length=4096,
    )

    assert "<｜Assistant｜><think>reasoning" in tokenizer.last_text
    assert "<｜Assistant｜></think><think>" not in tokenizer.last_text
    content_start = tokenizer.last_text.index(assistant_body)
    assert out["loss_mask"][content_start].item() == 1
    assert out["loss_mask"][content_start + len("<think>reasoning</think>")].item() == 1
