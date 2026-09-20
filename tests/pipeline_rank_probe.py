"""CPU-only distributed data protocol probe; does not instantiate a model."""

import json
import os
from datetime import timedelta
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from torchtitan.models.dspark_draft.data import PreparedTokens

from deepspec.pipeline.data import MooncakeFeatureLoader
from deepspec.pipeline.store import object_keys
from deepspec.pipeline.topology import consumer_dp
from deepspec.pipeline.runtime import Deadline, atomic_json


def data_contract(config, *, rank):
    dp, world = consumer_dp(config), config["consumer_world_size"]
    batch, count = config.get("samples_per_update", 4), len(config["samples"])
    steps = config.get("steps", count // batch)
    if (
        steps <= 0
        or batch != 4
        or batch % dp
        or count != steps * batch
        or [s["position"] for s in config["samples"]] != list(range(count))
    ):
        raise ValueError("Probe inputs must contain complete updates")
    if world != 4 * dp or not 0 <= rank < world:
        raise ValueError("Probe rank/world differs from TP4 training")
    dp_rank = rank // 4
    return {
        "steps": steps,
        "gas": batch // dp,
        "dp_rank": dp_rank,
        "positions": list(range(dp_rank, count, dp)),
        "native_cursor": count // dp,
        "sample_cursor": count,
        "context_length": max(s["length"] for s in config["samples"]),
    }


def main():
    config_path = os.environ["DEEPSPEC_PIPELINE_CONFIG"]
    config = json.loads(Path(config_path).read_text())
    timeout = config.get("timeouts_seconds", {}).get(
        "collective", config["timeout_seconds"]
    )
    Deadline.after(timeout)
    dist.init_process_group("gloo", timeout=timedelta(seconds=timeout))
    contract = data_contract(config, rank=dist.get_rank())
    assert dist.get_world_size() == config["consumer_world_size"]
    dp = consumer_dp(config)
    dp_rank = dist.get_rank() // (config["consumer_world_size"] // dp)
    loader = MooncakeFeatureLoader.Config(
        manifest=config["manifest_path"],
        plan_path=config["plan_path"],
        target_layer_ids=config["teacher"]["target_layer_ids"],
        hidden_size=config["teacher"]["hidden_size"],
        pipeline_config=config_path,
    ).build(
        dp_world_size=dp,
        dp_rank=dp_rank,
        max_context_length=contract["context_length"],
        num_tokens_per_batch=contract["context_length"],
        tokenizer=PreparedTokens.Config(vocab_size=128).build(tokenizer_path=""),
    )
    try:
        identity = {
            "rank": dist.get_rank(),
            "node_id": ray.get_runtime_context().get_node_id(),
            "run_id": config["run_id"],
            "plan_hash": config.get("plan_hash"),
        }
        if config.get("topology_plan_path"):
            from deepspec.pipeline.planning import TopologyPlan
            from tests.pipeline_collective_probe import rank_contract

            plan = TopologyPlan.from_dict(
                json.loads(Path(config["topology_plan_path"]).read_text())
            ).to_dict()
            native = rank_contract(
                plan,
                rank=dist.get_rank(),
                local_rank=int(os.environ["LOCAL_RANK"]),
                world=dist.get_world_size(),
                node_id=identity["node_id"],
            )
            assert native["positions"] == contract["positions"]
            assert native["native_cursor"] == contract["native_cursor"]
        gathered = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, identity)
        assert {i["rank"] for i in gathered} == set(range(dist.get_world_size()))
        assert {(i["run_id"], i["plan_hash"]) for i in gathered} == {
            (config["run_id"], config.get("plan_hash"))
        }
        iterator = iter(loader)
        observed = []
        for _ in range(contract["steps"]):
            # Preserve Titan's whole-GAS metadata fetch before computation.
            group = [next(iterator) for _ in range(contract["gas"])]
            for batch, labels in group:
                descriptor = batch.pop("_mooncake_features")
                position = descriptor["position"]
                features = loader.take_features(descriptor, "cpu")
                assert torch.equal(batch["input_ids"], labels)
                ray.get(
                    loader.buffer.acknowledge.remote(position, dist.get_rank()),
                    timeout=timeout,
                )
                dist.barrier()
                assert all(
                    loader.store.client.is_exist(key) == 0
                    for key in object_keys(descriptor["fields"])
                )
                assert bool(features["target_hidden_states"].eq(position + 1).all())
                assert bool(features["target_last_hidden_states"].eq(-position).all())
                observed.append(position)
        assert observed == contract["positions"]
        assert loader.cursor == contract["native_cursor"]
        assert 0 < loader.prefetch.peak_pending <= loader.prefetch.depth
        assert not loader.descriptors and not loader.prefetch.pending
        atomic_json(
            Path(config["result_prefix"] + f"-{dist.get_rank()}.json"),
            {**identity, **contract, "positions": observed, "cursor": loader.cursor},
        )
    finally:
        loader.close()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
