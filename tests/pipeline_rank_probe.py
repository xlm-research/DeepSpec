"""CPU-only distributed data protocol probe; does not instantiate a model."""

import json
import os
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from torchtitan.models.dspark_draft.data import PreparedTokens

from deepspec.pipeline.data import MooncakeFeatureLoader
from deepspec.pipeline.store import object_keys
from deepspec.pipeline.topology import consumer_dp


def main():
    dist.init_process_group("gloo")
    config_path = os.environ["DEEPSPEC_PIPELINE_CONFIG"]
    config = json.loads(Path(config_path).read_text())
    dp = consumer_dp(config)
    dp_rank = dist.get_rank() // (config["consumer_world_size"] // dp)
    loader = MooncakeFeatureLoader.Config(
        manifest=config["manifest_path"],
        plan_path=config["plan_path"],
        target_layer_ids=[1, 3],
        hidden_size=64,
        pipeline_config=config_path,
    ).build(
        dp_world_size=dp,
        dp_rank=dp_rank,
        max_context_length=16,
        num_tokens_per_batch=16,
        tokenizer=PreparedTokens.Config(vocab_size=128).build(tokenizer_path=""),
    )
    try:
        iterator = iter(loader)
        observed = []
        for _ in range(2):
            # Preserve Titan's whole-GAS metadata fetch before computation.
            group = [next(iterator) for _ in range(4 // dp)]
            for batch, labels in group:
                descriptor = batch.pop("_mooncake_features")
                position = descriptor["position"]
                features = loader.take_features(descriptor, "cpu")
                assert torch.equal(batch["input_ids"], labels)
                ray.get(loader.buffer.acknowledge.remote(position, dist.get_rank()))
                dist.barrier()
                assert all(
                    loader.store.client.is_exist(key) == 0
                    for key in object_keys(descriptor["fields"])
                )
                assert bool(features["target_hidden_states"].eq(position + 1).all())
                assert bool(features["target_last_hidden_states"].eq(-position).all())
                observed.append(position)
        assert observed == list(range(dp_rank, 8, dp))
        assert loader.cursor == 8 // dp
        assert loader.prefetch.peak_pending == 2
        assert not loader.descriptors and not loader.prefetch.pending
        Path(config["result_prefix"] + f"-{dist.get_rank()}.json").write_text(
            json.dumps({"positions": observed, "cursor": loader.cursor}) + "\n"
        )
    finally:
        loader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
