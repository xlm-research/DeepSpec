import os

import torch
from deepspec.data.target_cache_dataset import (
    CacheDataset,
    LocalCacheWriteSummary,
    LocalTargetCacheWriter,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    finalize_target_cache_indices,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)


def test_context_parallel_cache_roundtrip(tmp_path):
    output_dir = str(tmp_path)
    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir)
    summaries = []
    input_ids = torch.arange(8)
    mask = torch.ones(8, dtype=torch.uint8)

    for rank, (start, end) in enumerate(((0, 4), (4, 8))):
        rank_dir = os.path.join(tmp_dir, f"rank_{rank}")
        os.makedirs(rank_dir)
        writer = LocalTargetCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=1 << 20,
        )
        writer.write_sample(
            sample_id=0,
            context_start=start,
            input_ids=input_ids,
            attention_mask=mask,
            loss_mask=mask,
            target_hidden_states=torch.full(
                (end - start, 6), rank + 1, dtype=torch.bfloat16
            ),
            target_last_hidden_states=torch.full(
                (end - start, 2), rank + 1, dtype=torch.bfloat16
            ),
        )
        writer.close()
        summary = LocalCacheWriteSummary(
            global_rank=rank,
            context_parallel_rank=rank,
            source_sample_start=0,
            source_sample_end=1,
            num_local_samples=1,
            num_local_shards=len(writer.local_shard_files),
            local_shard_files=list(writer.local_shard_files),
        )
        atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))
        summaries.append(summary.to_json())

    shard_map, shards = build_global_target_cache_shard_map(summaries)
    for summary in summaries:
        rank = int(summary["global_rank"])
        rename_local_target_cache_shards(
            output_dir=output_dir,
            rank_dir=os.path.join(tmp_dir, f"rank_{rank}"),
            summary=summary,
            shard_map=shard_map,
        )
    num_samples, index_files = finalize_target_cache_indices(
        output_dir=output_dir,
        summaries=summaries,
        shard_map=shard_map,
        context_parallel_size=2,
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=[0, 1, 2],
        hidden_size=2,
        extra_fields={
            "cache_context_parallel_size": 2,
            "context_layout": "contiguous",
            "index_files": index_files,
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)

    for rank in range(2):
        dataset = CacheDataset(
            output_dir,
            context_parallel_size=2,
            context_parallel_rank=rank,
        )
        sample = dataset[0]
        assert sample["context_start"] == rank * 4
        assert sample["context_len"] == 4
        assert sample["input_ids"].tolist() == list(range(8))
        assert int(sample["target_hidden_states"][0, 0]) == rank + 1
        dataset.close()


def test_native_head_tail_cache_roundtrip(tmp_path):
    output_dir = str(tmp_path)
    tmp_dir = os.path.join(output_dir, "_tmp")
    os.makedirs(tmp_dir)
    summaries = []
    input_ids = torch.arange(7)
    mask = torch.ones(7, dtype=torch.uint8)

    for rank in range(2):
        rank_dir = os.path.join(tmp_dir, f"rank_{rank}")
        os.makedirs(rank_dir)
        writer = LocalTargetCacheWriter(
            rank_dir=rank_dir,
            max_shard_bytes=1 << 20,
            context_layout="native_head_tail",
        )
        writer.write_sample(
            sample_id=0,
            context_start=0,
            input_ids=input_ids,
            attention_mask=mask,
            loss_mask=mask,
            target_hidden_states=torch.full(
                (4, 6),
                rank + 1,
                dtype=torch.bfloat16,
            ),
            target_last_hidden_states=torch.full(
                (4, 2),
                rank + 1,
                dtype=torch.bfloat16,
            ),
        )
        writer.close()
        summary = LocalCacheWriteSummary(
            global_rank=rank,
            context_parallel_rank=rank,
            source_sample_start=0,
            source_sample_end=1,
            num_local_samples=1,
            num_local_shards=len(writer.local_shard_files),
            local_shard_files=list(writer.local_shard_files),
        )
        atomic_json_dump(
            summary.to_json(),
            os.path.join(rank_dir, "summary.json"),
        )
        summaries.append(summary.to_json())

    shard_map, shards = build_global_target_cache_shard_map(summaries)
    for summary in summaries:
        rank = int(summary["global_rank"])
        rename_local_target_cache_shards(
            output_dir=output_dir,
            rank_dir=os.path.join(tmp_dir, f"rank_{rank}"),
            summary=summary,
            shard_map=shard_map,
        )
    num_samples, index_files = finalize_target_cache_indices(
        output_dir=output_dir,
        summaries=summaries,
        shard_map=shard_map,
        context_parallel_size=2,
        context_layout="native_head_tail",
    )
    manifest = build_target_cache_manifest(
        num_samples=num_samples,
        shards=shards,
        target_layer_ids=[0, 1, 2],
        hidden_size=2,
        extra_fields={
            "cache_context_parallel_size": 2,
            "context_layout": "native_head_tail",
            "index_files": index_files,
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)

    for rank in range(2):
        dataset = CacheDataset(
            output_dir,
            context_parallel_size=2,
            context_parallel_rank=rank,
        )
        sample = dataset[0]
        assert sample["context_start"] == 0
        assert sample["context_len"] == 4
        assert sample["seq_len"] == 7
        assert int(sample["target_hidden_states"][0, 0]) == rank + 1
        dataset.close()
