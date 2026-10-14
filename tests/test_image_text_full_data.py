"""Full-data image-text behavior through the example preparation interface."""

import json
import os
import re
import runpy
import subprocess
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image


@pytest.fixture
def checkpoint():
    model = os.environ.get("TARGET_MODEL_PATH")
    if not model or not (Path(model) / "chat_template.jinja").is_file():
        pytest.skip("Set TARGET_MODEL_PATH to the local Qwen3.5 multimodal checkpoint")
    return model


def prepare_full(tmp_path, checkpoint, messages, *, max_length=4096, relative_output=False):
    Image.new("RGB", (32, 32), "red").save(tmp_path / "image.png")
    source = tmp_path / "source.jsonl"
    source.write_text(json.dumps({"messages": messages}) + "\n")
    output = tmp_path / "prepared"
    args = SimpleNamespace(
        source=str(source),
        media_root=str(tmp_path),
        model=checkpoint,
        indices="0",
        rows=16,
        max_length=max_length,
        output="prepared" if relative_output else str(output),
        full_data=True,
        expected_sources=1,
        min_loss_tokens=14,
    )
    script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/prepare_data.py"
    runpy.run_path(str(script))["prepare"](args)
    rows = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
    return rows, json.loads((output / "manifest.json").read_text())


def test_full_data_preparation_preserves_all_answers_without_smoke_repetition(tmp_path, checkpoint):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "The image is red and contains no other objects.",
            "reasoning_content": "I will inspect the color before answering.",
        },
        {"role": "user", "content": "Explain your previous answer in more detail."},
        {"role": "assistant", "content": "Every pixel in this picture is red, so it is uniform."},
    ]
    rows, manifest = prepare_full(tmp_path, checkpoint, messages)

    assert len(rows) == 1
    assert [m for m in rows[0]["messages"] if m["role"] == "assistant"] == [
        messages[1],
        messages[3],
    ]
    assert rows[0]["metadata"]["supervision_window"]["supervised_message_indices"] == [1, 3]
    assert manifest["source_count"] == 1
    assert manifest["window_count"] == 1
    assert manifest["sources"][0]["assistant_message_indices"] == [1, 3]
    assert "_repeat_" not in rows[0]["data_id"]


def test_full_data_windows_fit_expanded_limit_without_cutting_assistant_segments(
    tmp_path, checkpoint
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "old context " * 600},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "alpha " * 80,
            "reasoning_content": "Preserve the complete first reasoning.",
        },
        {"role": "user", "content": "Now continue."},
        {
            "role": "assistant",
            "content": "beta " * 80,
            "reasoning_content": "Preserve the complete second reasoning.",
        },
    ]
    rows, manifest = prepare_full(tmp_path, checkpoint, messages, max_length=256)

    assignments = []
    for row in rows:
        window = row["metadata"]["supervision_window"]
        assert window["expanded_tokens"] <= 255
        assignments.extend(window["supervised_message_indices"])
        retained = dict(zip(window["message_indices"], row["messages"], strict=True))
        for index in window["supervised_message_indices"]:
            assert retained[index] == messages[index]
    assert sorted(assignments) == [1, 3]
    assert any(w["context_reductions"] for w in manifest["windows"])


def test_full_data_mask_supervises_selected_segments_and_excludes_overlapping_history(
    tmp_path, checkpoint
):
    from transformers import AutoProcessor, AutoTokenizer

    from torchspec.data.renderers import RENDERER_REGISTRY
    from torchspec.data.utils import extract_media_urls

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "context " * 600},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {"role": "assistant", "content": "alpha " * 80, "reasoning_content": "First reasoning."},
        {"role": "user", "content": "Continue the previous answer."},
        {"role": "assistant", "content": "beta " * 80, "reasoning_content": "Second reasoning."},
    ]
    rows, _ = prepare_full(tmp_path, checkpoint, messages, max_length=256)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    renderer = RENDERER_REGISTRY.create("qwen3.5-full-data", tokenizer)
    observed_context = False
    for row in rows:
        window = row["metadata"]["supervision_window"]
        text = renderer.format(row["messages"], row.get("tools"))
        urls = (extract_media_urls(row["messages"]) or {}).get("images", [])
        images = [Image.open(url.removeprefix("file://")).convert("RGB") for url in urls]
        ids = processor(text=text, images=images or None, return_tensors="pt")["input_ids"][
            0
        ].tolist()
        mask = renderer.compute_loss_mask(ids, supervision_window=window)
        expanded_text = tokenizer.decode(ids, skip_special_tokens=False)
        offsets = tokenizer(expanded_text, add_special_tokens=False, return_offsets_mapping=True)
        assert offsets["input_ids"] == ids
        blocks = list(
            re.finditer(
                r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>",
                expanded_text,
                re.DOTALL,
            )
        )
        selected = [blocks[ordinal] for ordinal in window["supervised_assistant_ordinals"]]
        expected = [
            int(any(start >= block.start(1) and end <= block.end(1) for block in selected))
            for start, end in offsets["offset_mapping"]
        ]
        assert mask == expected
        observed_context |= len(selected) < len(blocks)
    assert observed_context


def test_full_data_preserves_later_system_role_and_order_without_changing_legacy_renderer(
    tmp_path, checkpoint
):
    from transformers import AutoTokenizer

    from torchspec.data.renderers import RENDERER_REGISTRY

    messages = [
        {"role": "system", "content": "INITIAL_SYSTEM_RULE"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "EARLY_ANSWER: this uniform picture is red.",
            "reasoning_content": "The initial answer precedes the later instruction.",
        },
        {"role": "system", "content": "LATER_SYSTEM_RULE"},
        {"role": "user", "content": "Answer again with the new instruction."},
        {
            "role": "assistant",
            "content": "LATE_ANSWER: the instruction now applies.",
            "reasoning_content": "Only the later answer can follow the new rule.",
        },
    ]
    rows, _ = prepare_full(tmp_path, checkpoint, messages)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    renderer = RENDERER_REGISTRY.create("qwen3.5-full-data", tokenizer)
    text = renderer.format(rows[0]["messages"])
    assert (
        text.index("EARLY_ANSWER")
        < text.index("<|im_start|>system\nLATER_SYSTEM_RULE")
        < text.index("LATE_ANSWER")
    )
    assert rows[0]["messages"][3] == messages[3]
    legacy = RENDERER_REGISTRY.create("qwen3.5-multimodal", tokenizer)
    with pytest.raises(Exception, match="System message must be at the beginning"):
        legacy.format(rows[0]["messages"])


def test_full_data_dry_run_selects_the_full_workflow_without_starting_training():
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [
            "bash",
            str(root / "examples/qwen3.8-27b-image-text-1node/run.sh"),
            "--dry-run",
            "--full-data",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=15,
    )
    assert "full_train.py" in result.stdout
    assert "torchspec.train_entry" not in result.stdout


def test_full_data_caches_remote_bytes_and_retains_gif_first_frame_semantics(
    tmp_path, checkpoint, monkeypatch
):
    import requests

    from vllm.multimodal import utils as media_utils

    frames = [Image.new("RGB", (32, 32), color) for color in ("red", "blue")]
    frames[0].save(tmp_path / "animation.gif", save_all=True, append_images=frames[1:])
    buffer = BytesIO()
    Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(buffer, format="PNG")
    payload = buffer.getvalue()
    url = "https://example.invalid/transparent.png"
    downloads = []

    def download(address, **kwargs):
        downloads.append(address)
        return SimpleNamespace(content=payload, raise_for_status=lambda: None)

    monkeypatch.setattr(requests, "get", download)
    fetch_image = media_utils.fetch_image

    def fetch(address, *args, **kwargs):
        if address == url:
            return Image.open(BytesIO(payload))
        return fetch_image(address, *args, **kwargs)

    monkeypatch.setattr(media_utils, "fetch_image", fetch)
    monkeypatch.chdir(tmp_path)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "animation.gif"}},
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        },
        {
            "role": "assistant",
            "content": "The first frame is red and the transparent image is white.",
            "reasoning_content": "Treat every input as a still image.",
        },
    ]
    rows, manifest = prepare_full(tmp_path, checkpoint, messages, relative_output=True)
    references = [block["image_url"]["url"] for block in rows[0]["messages"][0]["content"]]
    assert all(reference.startswith("file://") for reference in references)
    cached = Path(references[1].removeprefix("file://"))
    assert cached.read_bytes() == payload
    assert downloads == [url]
    decoded = [fetch_image(reference) for reference in references]
    images = [getattr(value, "media", value) for value in decoded]
    assert images[0].getpixel((0, 0)) == (255, 0, 0)
    assert images[1].getpixel((0, 0)) == (255, 255, 255)
    assert manifest["max_images"] == 3
    assert manifest["media"][url]["local_uri"] == references[1]


def test_full_data_loading_preserves_window_selection_for_target_prefill(tmp_path, checkpoint):
    from torchspec.data.dataset import load_conversation_dataset

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "context " * 600},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {"role": "assistant", "content": "alpha " * 80, "reasoning_content": "First reasoning."},
        {"role": "user", "content": "Continue the previous answer."},
        {"role": "assistant", "content": "beta " * 80, "reasoning_content": "Second reasoning."},
    ]
    rows, _ = prepare_full(tmp_path, checkpoint, messages, max_length=256)
    args = SimpleNamespace(
        train_data_path=str(tmp_path / "prepared/train.jsonl"),
        target_model_path=checkpoint,
        prompt_key="messages",
        renderer="qwen3.5-full-data",
        chat_template="qwen",
        max_seq_length=256,
        defer_tokenization=False,
        dynamic_loss_mask=True,
        last_turn_loss_only=False,
        train_with_decode=False,
        min_loss_tokens=14,
        num_proc=1,
        tokenization_batch_size=1,
        cache_dir=str(tmp_path / "cache"),
        inference_engine_type="vllm",
    )
    prompts = load_conversation_dataset(args)
    assert [p["data_id"] for p in prompts] == [r["data_id"] for r in rows]
    for prompt, row in zip(prompts, rows, strict=True):
        expected = row["metadata"]["supervision_window"]
        actual = prompt["metadata"]["supervision_window"]
        assert actual["supervised_message_indices"] == expected["supervised_message_indices"]
        assert actual["supervised_assistant_ordinals"] == expected["supervised_assistant_ordinals"]
        assert actual["expanded_tokens"] == expected["expanded_tokens"]
        assert "packed_loss_mask" not in prompt if prompt["multimodal_inputs"] else True


def test_oversized_user_context_is_shortened_once_for_a_group_of_complete_segments(
    tmp_path, checkpoint
):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "context " * 600},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {"role": "assistant", "content": "alpha " * 80, "reasoning_content": "First reasoning."},
        {"role": "user", "content": "Continue the previous answer."},
        {"role": "assistant", "content": "beta " * 80, "reasoning_content": "Second reasoning."},
    ]
    rows, _ = prepare_full(tmp_path, checkpoint, messages, max_length=384)
    assert len(rows) == 1
    window = rows[0]["metadata"]["supervision_window"]
    assert window["supervised_message_indices"] == [1, 3]
    assert window["expanded_tokens"] <= 383
    assert rows[0]["messages"][1] == messages[1]
    assert rows[0]["messages"][3] == messages[3]


def test_full_data_configuration_replaces_smoke_horizon_and_restoration(tmp_path, checkpoint):
    from torchspec.config.train_config import load_config

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image."},
                {"type": "image_url", "image_url": {"url": "image.png"}},
            ],
        },
        {
            "role": "assistant",
            "content": "The image is a uniform red square.",
            "reasoning_content": "Inspect its pixels before answering.",
        },
    ]
    prepare_full(tmp_path, checkpoint, messages, max_length=131072)
    script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/full_train.py"
    writer = runpy.run_path(str(script))["write_pass_config"]
    path = writer(
        tmp_path / "prepared/manifest.json",
        tmp_path / "prepared/train.jsonl",
        tmp_path / "run",
        tmp_path / "cache",
    )
    config = load_config(str(path))
    assert config.training.complete_data_pass
    assert config.training.num_train_steps == 1
    assert config.training.lr_total_steps == 1
    assert config.training.num_epochs == 1
    assert config.training.load_path is None
    assert config.training.ckpt_step is None
    assert config.model.initial_draft_model_path is None
    assert config.dataset.renderer == "qwen3.5-full-data"
    assert config.inference.vllm.extra_args.limit_mm_per_prompt.image == 1
    assert config.logging.report_to == "tensorboard"


def test_gpu_preflight_selects_distinct_windows_covering_changed_behavior_and_tail(tmp_path):
    from torchspec.controller.data_pass import CompleteDataPass

    rows = []
    for i in range(20):
        window = {
            "window_id": f"window_{i}",
            "source_id": f"source_{i}",
            "source_index": 100 + i,
            "supervised_message_indices": [1],
            "supervised_assistant_ordinals": [0],
            "assistant_count": 1,
            "expanded_tokens": 16,
            "supervised_tokens": 14,
            "image_count": 0,
        }
        rows.append(
            {
                "data_id": window["window_id"],
                "metadata": {"supervision_window": window},
                "messages": [
                    {"role": "user", "content": "Question"},
                    {"role": "assistant", "content": "Complete answer"},
                ],
            }
        )
    rows[0]["metadata"]["supervision_window"]["expanded_tokens"] = 131071
    rows[1]["metadata"]["supervision_window"]["image_count"] = 5
    rows[2]["messages"].append({"role": "system", "content": "Later instruction"})
    rows[3]["metadata"]["supervision_window"].update(
        assistant_count=2,
        supervised_assistant_ordinals=[1],
        supervised_message_indices=[3],
    )
    rows[4]["messages"][0]["content"] = [
        {"type": "image_url", "image_url": {"url": "file:///fixtures/animation.gif"}},
    ]
    rows[5]["messages"].insert(
        0,
        {
            "role": "tool",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": "file:///fixtures/full-data/media/remote.image"},
                },
            ],
        },
    )
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "source_count": 20,
                "window_count": 20,
                "windows": [row["metadata"]["supervision_window"] for row in rows],
            }
        )
    )
    script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/full_train.py"
    prepare = runpy.run_path(str(script))["prepare_preflight_inputs"]
    subset_data, subset_manifest = prepare(data, manifest, tmp_path / "preflight")
    chosen = [json.loads(line) for line in subset_data.read_text().splitlines()]
    ids = [row["data_id"] for row in chosen]
    assert len(ids) == len(set(ids)) == 17
    assert {f"window_{i}" for i in range(6)} <= set(ids)
    assert "not full-source acceptance" in json.loads(subset_manifest.read_text())["purpose"]
    CompleteDataPass(str(subset_manifest), 4, 4).validate_dataset(chosen)
