"""Prepare the immutable sample/update plan without constructing a GPU model."""

import argparse
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path

import torch

from .jsonl_source import JsonLineDataset
from .planning import input_identity, model_identity
from .preprocessing import preprocess_record


@dataclass(kw_only=True, slots=True)
class PreparationConfig:
    source_paths: list[str] = field(default_factory=list)
    epochs: int = 1
    chat_template: str = "qwen"
    min_loss_tokens: int = 1


def prepare_inputs(request):
    from transformers import AutoTokenizer

    from torchtitan.config.manager import ConfigManager
    from torchtitan.distributed.parallel_dims import ParallelDims

    config = ConfigManager().parse_args(request["recipe_args"])
    preparation = config.preparation
    if preparation is None or not preparation.source_paths:
        raise ValueError("The native recipe must define its preparation sources")
    if config.debug.seed is None:
        raise ValueError("Input preparation requires an explicit training seed")
    topology = ParallelDims.from_config(config.parallelism, int(request["workers"]))
    context = config.training.max_context_length
    micro_tokens = config.training.num_tokens_per_microbatch_per_dp_rank
    if micro_tokens != context:
        raise ValueError(
            "Producer integration requires one sample per logical DP microbatch"
        )
    dp_size = topology.dp_shard * topology.dp_replicate
    step_tokens = config.training.num_tokens_per_train_step
    if step_tokens < 0:
        step_tokens = micro_tokens * dp_size
    if step_tokens % (micro_tokens * dp_size):
        raise ValueError("Training tokens do not define complete optimizer updates")
    gas = step_tokens // (micro_tokens * dp_size)
    if topology.pp > 1 and gas % config.parallelism.num_pp_microbatches:
        raise ValueError("Logical GAS must contain complete pipeline schedules")
    global_batch = dp_size * gas
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    dataset = JsonLineDataset(preparation.source_paths, cache_dir=str(output / "index"))
    samples_per_epoch = len(dataset) // global_batch * global_batch
    if (
        not samples_per_epoch
        or config.training.steps * global_batch > samples_per_epoch * preparation.epochs
    ):
        raise ValueError("The dataset cannot supply the requested complete updates")
    tokenizer = AutoTokenizer.from_pretrained(
        config.hf_assets_path, local_files_only=True
    )
    batches = []
    wanted = config.training.steps * global_batch
    try:
        for epoch in range(preparation.epochs):
            generator = torch.Generator().manual_seed(config.debug.seed + epoch)
            order = torch.randperm(len(dataset), generator=generator)[
                :samples_per_epoch
            ]
            for sample_index in order.tolist():
                position = len(batches)
                if position == wanted:
                    break
                record = dataset[sample_index]
                conversations = record.get("packed_conversations")
                records = (
                    [{"conversations": value} for value in conversations]
                    if conversations
                    else [record]
                )
                pieces = []
                remaining = context
                for part in records:
                    piece = preprocess_record(
                        part, tokenizer, preparation.chat_template, remaining
                    )
                    pieces.append(piece)
                    remaining -= piece["input_ids"].numel()
                    if remaining <= 0:
                        break
                batch = {
                    key: torch.cat([piece[key] for piece in pieces]).unsqueeze(0)
                    for key in ("input_ids", "loss_mask")
                }
                if topology.pp > 1 and batch["input_ids"].shape[1] != context:
                    raise ValueError(
                        "The DSpark PP recipe requires fixed-length packed inputs"
                    )
                if batch["loss_mask"].count_nonzero() < preparation.min_loss_tokens:
                    raise ValueError(
                        f"Scheduled sample {sample_index} has insufficient supervision after truncation"
                    )
                batch_id = f"batch-{position // dp_size}-rank{position % dp_size}.pt"
                path = output / batch_id
                torch.save(batch, path)
                batches.append(
                    {
                        "id": batch_id,
                        "sample_id": f"epoch-{epoch}/sample-{sample_index}",
                        "position": position,
                        "epoch": epoch,
                        "input_path": str(path),
                        "input_identity": input_identity(batch),
                        "length": batch["input_ids"].shape[1],
                    }
                )
            if len(batches) == wanted:
                break
    finally:
        dataset.close()
    model = config.model_spec.model.hf_config
    plan = {
        "version": 1,
        "run_id": request["run_id"],
        "batches": batches,
        "global_batch_size": global_batch,
        "data_parallel_size": dp_size,
        "gradient_accumulation_steps": gas,
        "samples_per_epoch": samples_per_epoch,
        "training_steps": config.training.steps,
        "resolved_recipe": config.to_dict(),
        "producer_requirements": {
            **model_identity(config.initial_target_path),
            "target_layer_ids": model["target_layer_ids"],
            "hidden_size": model["hidden_size"],
            "activation_dtype": "bfloat16",
            "target_final_hidden_source": "full_model_final_norm_output",
        },
    }
    path = output / "input-plan.json"
    raw = json.dumps(plan, sort_keys=True).encode()
    temporary = path.with_suffix(".incomplete")
    with temporary.open("wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    if torch.cuda.is_initialized():
        raise RuntimeError("Input preparation unexpectedly initialized CUDA")
    return {
        "plan_path": str(path),
        "plan_identity": hashlib.sha256(raw).hexdigest(),
        "cuda_initialized": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request", type=Path)
    args = parser.parse_args()
    print(json.dumps(prepare_inputs(json.loads(args.request.read_text()))))
