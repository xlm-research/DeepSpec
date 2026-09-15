"""Export a committed draft's model weights on CPU without advancing training."""

import argparse
import json
from pathlib import Path
import time

import torch
import torch.distributed.checkpoint as dcp

from torchtitan.config import TORCH_DTYPE_MAP

from . import DSparkDraftModel
from .checkpoint import read_commit, write_marker


def weights_complete(output):
    index = output / "model.safetensors.index.json"
    try:
        names = (
            set(json.loads(index.read_text())["weight_map"].values())
            if index.is_file()
            else {"model.safetensors"}
        )
        return (output / "config.json").is_file() and all(
            (output / name).is_file() and (output / name).stat().st_size > 0
            for name in names
        )
    except (ValueError, KeyError):
        return False


def export_checkpoint(checkpoint, output_dir):
    started = time.monotonic()
    commit = read_commit(checkpoint)
    recipe = commit["resolved_recipe"]
    export_dtype = recipe["checkpoint"]["export_dtype"]
    output = Path(output_dir).resolve()
    marker = output / "export.json"
    if marker.is_file():
        try:
            previous = json.loads(marker.read_text())
        except ValueError:
            previous = None
        if previous and previous["checkpoint"] != commit:
            raise ValueError("Existing HF export belongs to a different checkpoint")
        if (
            previous
            and previous.get("export_dtype") == export_dtype
            and weights_complete(output)
        ):
            return previous
        marker.unlink()
    output.mkdir(parents=True, exist_ok=True)
    dtype = TORCH_DTYPE_MAP[recipe["training"]["dtype"]]
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        with torch.device("meta"):
            model = DSparkDraftModel.Config(
                hf_config=recipe["model_spec"]["model"]["hf_config"]
            ).build()
        model.to_empty(device="cpu")
    finally:
        torch.set_default_dtype(default_dtype)
    state = model.state_dict()
    dcp.load(state, checkpoint_id=checkpoint)
    model.load_state_dict(state, strict=True)
    model.to(dtype=TORCH_DTYPE_MAP[export_dtype])
    model.config.dtype = export_dtype
    model.save_pretrained(output, max_shard_size="5GB")
    model.config.architectures = ["Qwen3DSparkModel"]
    model.config.save_pretrained(output)
    if torch.cuda.is_initialized():
        raise RuntimeError("CPU export unexpectedly initialized CUDA")
    result = {
        "checkpoint": commit,
        "export_dtype": export_dtype,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_initialized": False,
    }
    write_marker(output, "export.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    print(json.dumps(export_checkpoint(args.checkpoint, args.output_dir)))
