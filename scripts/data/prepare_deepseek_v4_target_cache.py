#!/usr/bin/env python3
"""Precompute rank-sharded DeepSeek-V4 target features for training."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from transformers import AutoConfig, AutoTokenizer

from deepspec.data import ConversationCollator
from deepspec.data.jsonl_dataset import JsonLineDataset
from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_source_jsonl_fingerprints,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    compute_local_sample_range,
    finalize_target_cache_indices,
    load_local_cache_write_summary,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    validate_target_cache_identity,
    write_target_cache_manifest,
)
from deepspec.distributed import ParallelConfig, ParallelContext
from deepspec.modeling.target import DeepseekV4OnlineTarget
from deepspec.utils import (
    CustomJSONEncoder,
    get_git_sha,
    init_dist,
    is_global_main_process,
    load_config,
    main_process_first,
    parse_opts_to_config,
    print_on_global_main,
    print_on_local_main,
    seed_all,
)

os.environ["USE_TORCH"] = "true"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen DeepSeek-V4 target once per source record and "
            "persist its selected hidden states for later draft training."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--opts", action="append", default=[])
    parser.add_argument(
        "--train-data-path",
        action="append",
        required=True,
        help="Source JSONL path. Repeat to concatenate multiple files.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-shard-bytes", type=int, default=64 * 1024**3)
    args = parser.parse_args()
    config = parse_opts_to_config(args.opts, load_config(args.config))
    return args, config


def _target_parallel_context(*, config, parallel, world_size: int):
    merged = parallel.config.to_dict()
    merged.update(dict(config.train.get("target_parallel", {})))
    target_config = ParallelConfig.from_mapping(
        {"parallel": merged},
        world_size=world_size,
    )
    return parallel.with_sparse_config(target_config)


def _dummy_batch(*, device, context_parallel_size: int):
    # Every FSDP/EP rank must issue the same collectives even when its local
    # source range is exhausted or preprocessing rejects a record.
    length = max(int(context_parallel_size), 1)
    return {
        "input_ids": torch.zeros((1, length), dtype=torch.long, device=device),
        "attention_mask": torch.ones((1, length), dtype=torch.long, device=device),
        "loss_mask": torch.zeros((1, length), dtype=torch.long, device=device),
    }


def _reuse_existing_cache(
    *,
    output_dir: str,
    train_data_paths,
    config,
    context_parallel_size: int,
    stores_target_last_hidden_states: bool,
):
    manifest_path = os.path.join(output_dir, "manifest.json")
    manifest_present = [
        bool(os.path.isfile(manifest_path)) if is_global_main_process() else None
    ]
    dist.broadcast_object_list(manifest_present, src=0)
    if not manifest_present[0]:
        return False

    error = [None]
    if is_global_main_process():
        try:
            manifest = validate_target_cache_identity(
                cache_dir=output_dir,
                source_jsonl_paths=train_data_paths,
                target_model_name_or_path=config.model.target_model_name_or_path,
                target_layer_ids=config.model.target_layer_ids,
                chat_template=config.data.chat_template,
                max_length=int(config.data.max_length),
                context_parallel_size=context_parallel_size,
                stores_target_last_hidden_states=stores_target_last_hidden_states,
            )
            assert manifest["context_layout"] == "contiguous", (
                "DeepSeek-V4 requires contiguous CP cache shards."
            )
            assert int(manifest.get("min_loss_tokens", -1)) == int(
                config.data.get("min_loss_tokens", 1)
            ), "Minimum loss-token filtering does not match the cache."
        except (AssertionError, FileNotFoundError, OSError, ValueError) as exc:
            error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(error, src=0)
    if error[0] is not None:
        raise ValueError(
            "Existing target cache is stale or incompatible; use a different "
            f"--output-dir or remove it explicitly: {error[0]}"
        )
    print_on_global_main(f"Reusing offline DeepSeek-V4 target cache: {output_dir}")
    return True


def _write_manifest(
    *,
    output_dir: str,
    num_samples: int,
    index_files,
    shards,
    config,
    train_data_paths,
    context_parallel_size: int,
    parallel,
    target_parallel,
    stores_target_last_hidden_states: bool,
):
    target_config = AutoConfig.from_pretrained(
        config.model.target_model_name_or_path
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=config.model.target_layer_ids,
        hidden_size=int(target_config.hidden_size),
        extra_fields={
            "target_model_name_or_path": str(
                config.model.target_model_name_or_path
            ),
            "source_jsonl_paths": [str(path) for path in train_data_paths],
            "source_jsonl_fingerprints": build_source_jsonl_fingerprints(
                train_data_paths
            ),
            "chat_template": str(config.data.chat_template),
            "max_length": int(config.data.max_length),
            "min_loss_tokens": int(config.data.get("min_loss_tokens", 1)),
            "multimodal": False,
            "processor_class": None,
            "media_root": None,
            "media_uri_map": {},
            "target_context_parallel_size": int(context_parallel_size),
            "cache_context_parallel_size": int(context_parallel_size),
            "context_layout": "contiguous",
            "index_files": [str(path) for path in index_files],
            "target_context_parallel_implementation": "deepseek_v4_ring",
            "target_fsdp_size": int(parallel.config.fsdp_shard_size),
            "target_expert_parallel_size": int(
                target_parallel.expert_parallel_size
            ),
            "target_micro_chunk_size": 0,
            "target_cache_cpu_offload": False,
            "stores_target_last_hidden_states": bool(
                stores_target_last_hidden_states
            ),
            "project_name": str(config.get("project_name", "")),
            "exp_name": str(config.get("exp_name", "")),
            "git_sha": str(get_git_sha()),
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)


def main(local_rank: int):
    cli_args, config = parse_args()
    train_data_paths = [os.path.abspath(path) for path in cli_args.train_data_path]
    output_dir = os.path.abspath(cli_args.output_dir)
    seed_all(int(config.seed))
    device, global_rank, world_size = init_dist(local_rank)

    parallel_config = ParallelConfig.from_mapping(
        config.train,
        world_size=world_size,
    )
    if int(parallel_config.tp) != 1:
        raise NotImplementedError(
            "Offline DeepSeek-V4 target-cache generation currently requires TP=1."
        )
    if int(config.train.local_batch_size) != 1:
        raise ValueError(
            "Offline DeepSeek-V4 target-cache generation requires "
            "train.local_batch_size=1."
        )
    parallel = ParallelContext.build(parallel_config)
    target_parallel = _target_parallel_context(
        config=config,
        parallel=parallel,
        world_size=world_size,
    )
    context_parallel_size = int(parallel.context_parallel_size)
    stores_target_last_hidden_states = bool(
        config.data.get("store_target_last_hidden_states", True)
    )

    print_on_local_main(
        json.dumps(
            {
                "output_dir": output_dir,
                "source_jsonl_paths": train_data_paths,
                "target_layer_ids": list(config.model.target_layer_ids),
                "stores_target_last_hidden_states": (
                    stores_target_last_hidden_states
                ),
                "draft_parallel": parallel.config.to_dict(),
                "target_parallel": target_parallel.config.to_dict(),
            },
            indent=2,
            cls=CustomJSONEncoder,
        )
    )

    if _reuse_existing_cache(
        output_dir=output_dir,
        train_data_paths=train_data_paths,
        config=config,
        context_parallel_size=context_parallel_size,
        stores_target_last_hidden_states=stores_target_last_hidden_states,
    ):
        dist.barrier()
        dist.destroy_process_group()
        return

    prepare_error = [None]
    if is_global_main_process():
        try:
            prepare_target_cache_output_dir(output_dir)
        except (FileExistsError, OSError) as exc:
            prepare_error[0] = f"{type(exc).__name__}: {exc}"
    dist.broadcast_object_list(prepare_error, src=0)
    if prepare_error[0] is not None:
        raise RuntimeError(
            "Cannot initialize offline target cache output: "
            f"{prepare_error[0]}"
        )
    rank_dir = os.path.join(output_dir, "_tmp", f"rank_{global_rank}")
    os.makedirs(rank_dir, exist_ok=True)

    with main_process_first():
        dataset = JsonLineDataset(data_paths=train_data_paths)
    source_num_samples = len(dataset)
    local_start, local_end = compute_local_sample_range(
        num_samples=source_num_samples,
        rank=parallel.data_parallel_rank,
        world_size=parallel.data_parallel_size,
    )
    local_total_samples = local_end - local_start
    local_indices = list(range(local_start, local_end))
    max_local_steps = (
        source_num_samples + parallel.data_parallel_size - 1
    ) // parallel.data_parallel_size
    if len(local_indices) < max_local_steps:
        padding_index = local_indices[0] if local_indices else 0
        local_indices.extend(
            [padding_index] * (max_local_steps - len(local_indices))
        )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.target_model_name_or_path
    )
    collator = ConversationCollator(
        tokenizer=tokenizer,
        chat_template=config.data.chat_template,
        max_length=int(config.data.max_length),
        min_loss_tokens=int(config.data.get("min_loss_tokens", 1)),
    )
    dataloader = DataLoader(
        Subset(dataset, local_indices),
        batch_size=1,
        collate_fn=collator,
        num_workers=int(cli_args.num_workers),
        pin_memory=True,
        drop_last=False,
    )
    target = DeepseekV4OnlineTarget(
        model_name_or_path=config.model.target_model_name_or_path,
        target_layer_ids=config.model.target_layer_ids,
        topology=target_parallel,
        device=device,
        rank_local_cache_dir=rank_dir,
    )
    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(cli_args.max_shard_bytes),
        max_queue_size=1,
    )

    processed = 0
    try:
        for step, batch in enumerate(dataloader):
            is_padding = step >= local_total_samples
            should_write = batch is not None and not is_padding
            if batch is None:
                batch = _dummy_batch(
                    device=device,
                    context_parallel_size=context_parallel_size,
                )
            else:
                batch = {
                    key: value.to(device, non_blocking=True)
                    for key, value in batch.items()
                }
            result = target.forward_training_batch(batch)
            if should_write:
                writer.write_sample(
                    context_start=int(result["context_start"][0].item()),
                    input_ids=result["input_ids"][0],
                    attention_mask=torch.ones_like(result["input_ids"][0]),
                    loss_mask=result["loss_mask"][0],
                    target_hidden_states=result["target_hidden_states"][0],
                    target_last_hidden_states=(
                        result["target_last_hidden_states"][0]
                        if stores_target_last_hidden_states
                        else None
                    ),
                )
                processed += 1
                if processed % 10 == 0 or processed == local_total_samples:
                    print(
                        f"[offline-cache rank {global_rank}] "
                        f"{processed}/{local_total_samples} samples",
                        flush=True,
                    )
    finally:
        try:
            writer.close()
        finally:
            target.close()
            dataset.close()

    summary = LocalCacheWriteSummary(
        global_rank=global_rank,
        context_parallel_rank=parallel.context_parallel_rank,
        source_sample_start=local_start,
        source_sample_end=local_end,
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
    dist.barrier()

    summaries = None
    shard_map = None
    shards = None
    if is_global_main_process():
        summaries = [
            load_local_cache_write_summary(
                os.path.join(output_dir, "_tmp", f"rank_{rank}")
            )
            for rank in range(world_size)
        ]
        shard_map, shards = build_global_target_cache_shard_map(summaries)
    payload = [shard_map]
    dist.broadcast_object_list(payload, src=0)
    shard_map = payload[0]
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=summary.to_json(),
        shard_map=shard_map,
    )
    dist.barrier()

    if is_global_main_process():
        assert summaries is not None and shards is not None
        num_samples, index_files = finalize_target_cache_indices(
            output_dir=output_dir,
            summaries=summaries,
            shard_map=shard_map,
            context_parallel_size=context_parallel_size,
        )
        _write_manifest(
            output_dir=output_dir,
            num_samples=num_samples,
            index_files=index_files,
            shards=shards,
            config=config,
            train_data_paths=train_data_paths,
            context_parallel_size=context_parallel_size,
            parallel=parallel,
            target_parallel=target_parallel,
            stores_target_last_hidden_states=stores_target_last_hidden_states,
        )
        cleanup_target_cache_tmp_dir(output_dir)
        print_on_global_main(
            f"Prepared offline DeepSeek-V4 target cache at {output_dir}: "
            f"{num_samples}/{source_num_samples} valid samples."
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    if "LOCAL_RANK" in os.environ:
        main(int(os.environ["LOCAL_RANK"]))
    else:
        torch.multiprocessing.spawn(main, nprocs=torch.cuda.device_count())
