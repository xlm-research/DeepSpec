"""Real image-text prefill, expanded supervision and Mooncake capture validation."""

import argparse
import json
import os
import queue
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

from torchspec.config.mooncake_config import MooncakeConfig
from torchspec.config.train_config import config_to_flat_args, load_config
from torchspec.data.dataset import load_conversation_dataset
from torchspec.data.renderers import RENDERER_REGISTRY
from torchspec.data.utils import DataCollatorWithPadding, unpack_loss_mask
from torchspec.inference.engine.vllm_engine import VllmEngine
from torchspec.training.data_fetcher import MooncakeDataFetcher, TrainSample
from torchspec.transfer.mooncake import EagleMooncakeStore, MooncakeMaster


def read_capture(store, result):
    """Use the production consumer, including TP-only per-layer manifests."""
    pending = queue.Queue()
    pending.put(
        TrainSample(
            **{
                key: result.get(key)
                for key in (
                    "mooncake_key",
                    "tensor_shapes",
                    "tensor_dtypes",
                    "packed_loss_mask",
                    "metadata",
                    "data_id",
                )
            }
        )
    )
    fetcher = MooncakeDataFetcher(pending, store, DataCollatorWithPadding(), torch.device("cpu"))
    length = result["tensor_shapes"]["input_ids"][0]
    return {
        key: value.squeeze(0)[:length]
        for key, value in next(iter(fetcher)).items()
        if isinstance(value, torch.Tensor)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data")
    parser.add_argument("--output")
    parser.add_argument("--legacy-text-only", action="store_true")
    cli = parser.parse_args()
    output = Path(
        cli.output
        or (
            "outputs/qwen3.8-27b-image-text-1node-text-prefill"
            if cli.legacy_text_only
            else "outputs/qwen3.8-27b-image-text-1node-prefill"
        )
    )
    output.mkdir(parents=True, exist_ok=True)
    recipe = Path(__file__).with_name("config.yaml")
    if cli.legacy_text_only:
        recipe = recipe.parent.parent / "qwen3.8-27b-1node" / "config.yaml"
        fixture = output / "legacy_text.jsonl"
        fixture.write_text(
            json.dumps(
                {
                    "data_id": "legacy_text_regression",
                    "conversations": [
                        {"role": "user", "content": "Describe what a text regression verifies."},
                        {
                            "role": "assistant",
                            "content": "<think>Historical reasoning remains context.</think> An earlier answer.",
                        },
                        {
                            "role": "user",
                            "content": "Describe why preserving the existing entry point matters.",
                        },
                        {
                            "role": "assistant",
                            "content": "<think>The original template and supervision defaults must remain usable.</think> Existing text callers should continue to receive model features and complete a one token request using the same documented interface.",
                        },
                    ],
                }
            )
            + "\n"
        )
        data = cli.data or str(fixture)
    else:
        data = cli.data or "cache/qwen3.8-27b-image-text-1node/data/prefill.jsonl"
    args = config_to_flat_args(load_config(str(recipe)))
    args.train_data_path = str(Path(data).resolve())
    args.output_dir = str(output)
    args.num_proc = 1
    model_config = AutoConfig.from_pretrained(args.target_model_path, trust_remote_code=True)
    hidden_size = getattr(model_config, "text_config", model_config).hidden_size
    layer_count = len(args.aux_hidden_states_layers)
    renderer = RENDERER_REGISTRY.create(
        "qwen3.5-multimodal",
        AutoTokenizer.from_pretrained(args.target_model_path, trust_remote_code=True),
    )
    prompts = load_conversation_dataset(args)
    # Include a text-only request through the same public prefill interface.
    if not cli.legacy_text_only:
        prompts.append(
            {
                "data_id": "text_regression",
                "input_ids": torch.tensor([100, 200, 300]),
                "multimodal_inputs": None,
            }
        )
    master = MooncakeMaster()
    info = master.start(master.find_free_port(51135), master.find_free_port(8763))
    host = info["master_addr"].split(":")[0]
    config = MooncakeConfig(
        local_hostname=host,
        master_server_address=info["master_addr"],
        metadata_server=f"http://{host}:{info['metadata_port']}/metadata",
        global_segment_size="16GB",
        local_buffer_size="4GB",
        max_seq_len=131072,
        hidden_dim=hidden_size,
        num_aux_layers=layer_count,
    )
    engine = VllmEngine(args, rank=0, num_gpus_per_engine=4)
    store = EagleMooncakeStore(config)
    report = []
    try:
        engine.init(config)
        store.setup(device="cpu")
        for prompt in prompts:
            results = engine.generate(
                [prompt["data_id"]],
                [prompt["input_ids"]],
                packed_loss_mask_list=[prompt.get("packed_loss_mask")],
                multimodal_inputs=[prompt.get("multimodal_inputs")],
            )
            if len(results) != 1:
                raise RuntimeError("Prefill did not return exactly one complete capture")
            result = results[0]
            if len(result["prefill_completion_token_ids"]) != 1:
                raise RuntimeError("Prefill must generate exactly one token")
            tensors = read_capture(store, result)
            length = result["seq_len"]
            assert tuple(tensors["hidden_states"].shape) == (length, layer_count * hidden_size)
            assert tuple(tensors["last_hidden_states"].shape) == (length, hidden_size)
            assert tensors["input_ids"].tolist() == result["input_ids_list"]
            assert torch.isfinite(tensors["hidden_states"]).all()
            assert torch.isfinite(tensors["last_hidden_states"]).all()
            mask = tensors["loss_mask"]
            assert len(mask) == length
            if prompt.get("multimodal_inputs"):
                assert length > len(prompt["input_ids"])
                assert mask.sum() > 0
                for token in renderer.visual_token_ids:
                    assert mask[tensors["input_ids"] == token].sum() == 0
            if cli.legacy_text_only:
                assert torch.equal(mask, unpack_loss_mask(prompt["packed_loss_mask"]))
            entry = {
                "data_id": prompt["data_id"],
                "prompt_tokens": length,
                "completion_tokens": len(result["prefill_completion_token_ids"]),
                "supervised_tokens": int(mask.sum()),
                "hidden_states_shape": list(tensors["hidden_states"].shape),
                "last_hidden_states_shape": list(tensors["last_hidden_states"].shape),
                "all_finite": True,
                "capture_slots": engine.aux_hidden_state_layer_ids,
            }
            torch.save(
                {"input_ids": tensors["input_ids"], "loss_mask": mask},
                output / f"{prompt['data_id']}_tokens.pt",
            )
            report.append(entry)
            del tensors
            (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(entry), flush=True)
    finally:
        engine.shutdown()
        master.shutdown()


if __name__ == "__main__":
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    main()
