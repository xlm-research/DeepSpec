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


def test_preflight_selects_actual_gif_and_cached_images_despite_text_mentions(tmp_path):
    script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/full_train.py"
    select = runpy.run_path(str(script))["prepare_preflight_inputs"]
    cached = "file:///cache/full-data/media/remote.image"
    rows = []
    for index in range(20):
        uri = "file:///image.gif" if index == 18 else cached if index == 19 else "file:///image.png"
        window = {
            "source_id": f"source_{index}",
            "source_index": index,
            "window_id": f"window_{index}",
            "expanded_tokens": 2000 if index == 0 else 1000,
            "image_count": 1,
            "assistant_count": 1,
            "supervised_assistant_ordinals": [0],
            "supervised_message_indices": [1],
        }
        rows.append(
            {
                "data_id": window["window_id"],
                "metadata": {"supervision_window": window},
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Discuss .gif and /full-data/media/ paths."},
                            {"type": "image_url", "image_url": {"url": uri}},
                        ],
                    },
                    {"role": "assistant", "content": "An image."},
                ],
            }
        )
    data = tmp_path / "train.jsonl"
    data.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {"source_count": 20, "media": {"https://example.org/image": {"local_uri": cached}}}
        )
    )
    selected, _ = select(data, manifest, tmp_path / "preflight")
    ids = {json.loads(line)["data_id"] for line in selected.read_text().splitlines()}
    assert len(ids) == 17
    assert {"window_18", "window_19"} <= ids


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
    from transformers import AutoTokenizer

    from torchspec.data.renderers import RENDERER_REGISTRY

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    native_template = tokenizer.get_chat_template()
    renderer = RENDERER_REGISTRY.create("qwen3.5-full-data", tokenizer)
    ids, mask = renderer.render(rows[0]["messages"], max_seq_length=4096)
    supervised = tokenizer.decode([token for token, keep in zip(ids, mask, strict=True) if keep])
    assert messages[1]["reasoning_content"] in supervised
    assert messages[1]["content"] in supervised
    assert messages[3]["content"] in supervised
    assert tokenizer.get_chat_template() == native_template


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
    prefill_script = Path(__file__).parents[1] / "examples/qwen3.8-27b-image-text-1node/prefill.py"
    validate_capture = runpy.run_path(str(prefill_script))["validate_supervision_capture"]
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
        validate_capture(tokenizer, ids, mask, row)
        if len(selected) < len(blocks):
            context = next(block for block in blocks if block not in selected)
            context_token = next(
                i
                for i, (start, end) in enumerate(offsets["offset_mapping"])
                if start >= context.start(1) and end <= context.end(1)
            )
            wrong_mask = list(mask)
            wrong_mask[context_token] = 1
            with pytest.raises(AssertionError, match="independent assistant spans"):
                validate_capture(tokenizer, ids, wrong_mask, row)
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
    assert config.inference.inference_batch_size == 1
    assert config.inference.vllm.extra_args.max_num_seqs == 1
    assert config.inference.vllm.mem_fraction_static == 0.7
    assert (
        json.loads(config.training.train_env_vars).get("PYTORCH_CUDA_ALLOC_CONF")
        == "expandable_segments:True"
    )
    # Six 5120-wide BF16 feature layers plus per-layer input IDs for each
    # queued optimizer window, pooled result, and in-flight target request.
    capture_bytes = 131072 * 6 * (5120 * 2 + 8)
    resident_windows = (
        16 + config.inference.max_sample_pool_size + config.inference.inference_batch_size
    )
    segment_bytes = int(config.mooncake.global_segment_size.removesuffix("GB")) * 1024**3
    assert segment_bytes > resident_windows * capture_bytes
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


def test_loop_full_data_configuration_keeps_execution_features_and_postnorm(tmp_path, monkeypatch):
    from torchspec.config.train_config import load_config

    root = Path(__file__).parents[1]
    target = "/mnt/afs_agents/yankuihao/model_ckpts/1009_9b_mhc4_loop3_from50_zxh_glmv25_all16_16nodes/ckpt-225-bf16"
    monkeypatch.setenv("TARGET_MODEL_PATH", target)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "model": target,
                "window_count": 17,
                "min_loss_tokens": 14,
                "max_length": 131072,
                "max_images": 20,
            }
        )
    )
    writer = runpy.run_path(str(root / "examples/qwen3.8-27b-image-text-1node/full_train.py"))[
        "write_pass_config"
    ]
    path = writer(
        manifest,
        tmp_path / "train.jsonl",
        tmp_path / "output",
        tmp_path / "cache",
        base_config_path=root / "examples/qwen3.5-loop-mhc-image-text-dspark-1-node/config.yaml",
    )
    config = load_config(str(path))
    assert config.training.num_train_steps == config.training.lr_total_steps == 2
    assert config.inference.vllm.direct_hidden_state_capture
    assert not config.inference.last_hidden_states_prenorm
    assert config.inference.aux_hidden_states_layers == [3, 27, 51, 75, 78]
    assert not config.inference.vllm.extra_args.language_model_only
    assert config.inference.vllm.extra_args.limit_mm_per_prompt.image == 20
    assert config.debug.verify_parameter_updates
    assert Path(config.model.draft_model_config).name == "qwen3_5_loop_mhc_dspark.json"


def test_local_acceptance_matches_weighted_distribution_overlap_and_is_replotable(tmp_path):
    from torchspec.utils.dspark_metrics import LocalDSparkMetrics, plot_acceptance_curve

    target = [[0.6, 0.3, 0.1], [0.1, 0.1, 0.8]]
    draft = [[0.1, 0.2, 0.7], [0.1, 0.2, 0.7]]
    weights = [1.0, 0.25]
    overlap = sum(
        w * sum(min(p, q) for p, q in zip(ps, qs, strict=True))
        for w, ps, qs in zip(weights, target, draft, strict=True)
    ) / sum(weights)
    l1 = sum(
        w * sum(abs(p - q) for p, q in zip(ps, qs, strict=True))
        for w, ps, qs in zip(weights, target, draft, strict=True)
    ) / sum(weights)
    logger = LocalDSparkMetrics(str(tmp_path), l1_alpha=0.9)
    logger.record(
        {
            "train/ce_loss": 3.0,
            "train/l1_loss": l1,
            "train/confidence_loss": 0.5,
            "train/avg_acc": 0.99,
        },
        1,
    )
    logger.finalize(1)
    records = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert records[0]["estimated_training_acceptance"] == pytest.approx(overlap)
    assert records[0]["estimated_training_acceptance"] != pytest.approx(0.99)
    plot_acceptance_curve(records, tmp_path / "replot.png")
    for name in ("acceptance.png", "loss.png", "replot.png"):
        with Image.open(tmp_path / name) as image:
            image.verify()


@pytest.mark.parametrize("l1,weight", [(-0.1, 0.9), (2.1, 0.9), (float("nan"), 0.9), (0.0, 0.0)])
def test_acceptance_rejects_unmeasured_or_invalid_distribution_loss(l1, weight):
    from torchspec.utils.dspark_metrics import estimated_training_acceptance

    with pytest.raises(ValueError):
        estimated_training_acceptance({"l1_loss": l1, "loss_weights": {"l1_loss": weight}})
